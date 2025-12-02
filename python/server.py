import logging
import re
from pathlib import Path
from typing import Iterator, OrderedDict

import grpc
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from circular_external_samples_buffer import CircularExternalSamplesBuffer
from generated import training_pb2
from generated import training_pb2_grpc
from generated.training_pb2 import SendTrainingDataRequest
from neural.chess_model_bert_cls_v2 import ThreePlayerChessformerBertV2
from neural.chess_model_hybrid import ThreePlayerHybrid
from neural.chess_model_shaw import ThreePlayerChessformerShaw
from neural.model_factory import model_factory
from neural.model_wrapper import ModelWrapper
from server_settings import ServerSettings, settings_not_set_error


class TrainingCoordinatorService(training_pb2_grpc.TrainingCoordinatorServicer):
    settings: ServerSettings | None = None
    samples_buffer: CircularExternalSamplesBuffer | None = None
    model: ModelWrapper | None = None
    device: torch.device
    train_loader: DataLoader | None = None
    current_batch_size = -1
    summary_writer: SummaryWriter | None = None
    total_training_steps: int = 0
    scaler: GradScaler | None = None
    scheduler: optim.lr_scheduler.LRScheduler | None = None

    weight_decay: float = 1.0e-2
    policy_loss_weight: float = 1.0
    z_value_loss_weight: float = 10.0
    q_value_loss_weight: float = 5.0

    def SetInitialSettings(self, request: training_pb2.SetSettingsRequest, context):
        """
        This method handles a standard unary RPC.
        `request` is a single `SetSettingsRequest` object.
        """
        if self.settings is not None:
            raise grpc.RpcError(
                grpc.StatusCode.ALREADY_EXISTS,
                "Initial settings have already been set. Cannot set them again."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.settings = ServerSettings(request)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Using device '{self.device}' and settings: {self.settings}")

        use_amp = (self.device.type == 'cuda')

        # Start init_scale with 4096 to avoid early under-flows
        self.scaler = GradScaler(init_scale=2. ** 12, enabled=use_amp)
        if use_amp:
            logging.info("Enabling Automatic Mixed Precision (AMP).")

        latest_model_version: None | int = None

        net = model_factory(request.model_key)

        logging.info(
            f"State shape: '{net.state_shape()}', Policy shape: '{net.policy_shape()}', Value shape: '{net.value_shape()}', Aux features shape: '{(request.num_of_aux_features,)}'.")

        self.samples_buffer = CircularExternalSamplesBuffer(
            capacity=self.settings.max_samples,
            persistence_path=self.settings.samples_path,
            state_shape=net.state_shape(),
            policy_shape=net.policy_shape(),
            value_shape=net.value_shape(),
            aux_features_shape=(request.num_of_aux_features,),
            is_resuming=request.current_training_step > 0,
        )

        self.total_training_steps = request.current_training_step

        net.to(self.device)
        net.pre_onnx_export()

        # Set losses and weight decay from request
        self.weight_decay = request.weight_decay
        self.policy_loss_weight = request.policy_loss_weight
        self.z_value_loss_weight = request.z_value_loss_weight
        self.q_value_loss_weight = request.q_value_loss_weight

        logging.info(
            f"Set L2 to {self.weight_decay}, PI weight to {self.policy_loss_weight}, V Z weight to {self.z_value_loss_weight}, V Q weight to {self.q_value_loss_weight}.")

        decay_params = []
        no_decay_params = []
        no_decay_params_names = []

        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue

            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
                no_decay_params_names.append(name)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        logging.info(
            f"Not applying weight decay to {len(no_decay_params)} parameters ({', '.join(no_decay_params_names)}).")

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=request.learning_rate)

        using_torch_compile = False

        if self.device == torch.device("cuda"):
            net = torch.compile(net, mode="default", fullgraph=True)
            logging.info("Compiled the model for CUDA with max-autotune and fullgraph.")
            using_torch_compile = True

        for f in self.settings.pytorch_dir.glob("*.pt"):
            if match := re.match(r"(\d+)\.pt", f.name):
                version = int(match.group(1))
                if latest_model_version is None or version > latest_model_version:
                    latest_model_version = version

        if latest_model_version is not None:
            model_path = self.settings.pytorch_dir / f"{latest_model_version}.pt"
            optimizer_path = self.settings.optimizer_dir / f"{latest_model_version}.pt"
            logging.info(f"Loading existing model from {model_path} and optimizer from {optimizer_path}")

            if not model_path.exists() or not optimizer_path.exists():  # Sanity check
                raise grpc.RpcError(
                    grpc.StatusCode.NOT_FOUND,
                    f"Model or optimizer file does not exist at {model_path} or {optimizer_path}."
                )

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

            if not using_torch_compile:
                if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_key = k.replace("_orig_mod.", "", 1)  # strip only the first occurrence
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict

            net.load_state_dict(state_dict)

            optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))

            optimizer.param_groups[0]['lr'] = request.learning_rate
            optimizer.param_groups[1]['lr'] = request.learning_rate
            optimizer.param_groups[0]['weight_decay'] = self.weight_decay
            optimizer.param_groups[1]['weight_decay'] = 0.0

            logging.info(f"Manually forced optimizer learning rate to: {request.learning_rate}")

            self.model = ModelWrapper(model=net, optimizer=optimizer, version=latest_model_version)

        else:
            logging.info(f"Initializing new model with #parameters: {sum(p.numel() for p in net.parameters()):,}")
            logging.info(f"Learning rate: {request.learning_rate}")

            self.model = ModelWrapper(
                model=net,
                optimizer=optimizer,
                version=0
            )

            self.model.save_checkpoint(
                pytorch_dir=self.settings.pytorch_dir,
                optimizer_dir=self.settings.optimizer_dir
            )

        warmup_steps = request.warmup_steps
        logging.info(f"Using {warmup_steps} warmup steps for the scheduler.")

        self.scheduler = optim.lr_scheduler.LinearLR(
            self.model.optimizer,
            start_factor=0.01,  # Start at 1% of the full learning rate
            end_factor=1.0,
            total_iters=warmup_steps
        )

        logging.info(f"Initialized LinearLR warmup scheduler with {warmup_steps} steps.")

        onnx_path = self.model.save_to_onnx(self.settings.onnx_dir, batch_size=self.settings.batch_size)
        logging.info(f"Model saves to ONNX at {onnx_path} with batch size {self.settings.batch_size}.")
        self.summary_writer = SummaryWriter(self.settings.tensorboard_dir)

        if self.total_training_steps > 0:
            logging.info(f"Resuming from step {self.total_training_steps}. Fast-forwarding scheduler...")
            for _ in range(self.total_training_steps):
                self.scheduler.step()
            logging.info(f"Scheduler fast-forwarded. Current LR: {self.model.optimizer.param_groups[0]['lr']:.2e}")

        return training_pb2.InitialSettingsResponse(
            model_path=onnx_path.as_posix(),
        )

    def SendTrainingData(self, request_iterator: Iterator[SendTrainingDataRequest], context):
        if self.settings is None or self.samples_buffer is None:
            return settings_not_set_error()

        logging.info("Client connected, beginning to receive training data stream...")
        samples_in_this_stream = 0
        for request in request_iterator:
            update = request.update
            if not update:
                logging.error("Received a request with an empty sample field.")
                continue
            samples_in_this_stream += self.samples_buffer.load_specific_file(Path(update.parquet_file_path))
        logging.info(
            f"Stream finished gracefully. Total samples received: {samples_in_this_stream}, total in memory: {len(self.samples_buffer)}.")
        return training_pb2.SendTrainingDataResponse(
            samples_loaded_in_stream=samples_in_this_stream,
            total_samples_in_memory=len(self.samples_buffer),
        )

    def TrainModel(self, request: training_pb2.TrainModelRequest, context):
        """
        This method handles a standard unary RPC.
        `request` is a single `TrainModelRequest` object.
        """
        if self.settings is None or self.samples_buffer is None or self.model is None:
            return settings_not_set_error()

        if self.train_loader is None or self.current_batch_size != request.batch_size:
            start_time = time.time()
            self.train_loader = DataLoader(
                dataset=self.samples_buffer,
                batch_size=request.batch_size,
                shuffle=True,
                num_workers=0,
                persistent_workers=False,
                pin_memory=self.device.type == 'cuda',
                drop_last=True
            )
            self.current_batch_size = request.batch_size
            logging.info(f"New DataLoader created in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        self.model.model.train()
        non_block = True if self.device.type == 'cuda' else False
        logging.info(f"Starting training for {request.epochs} epochs with batch size {request.batch_size}.")

        pi_weight = self.policy_loss_weight
        values_z_weight = self.z_value_loss_weight
        values_q_weight = self.q_value_loss_weight

        np.set_printoptions(precision=2)

        uncompiled_model = self.model.model._orig_mod if hasattr(self.model.model, '_orig_mod') else self.model.model

        for epoch in range(request.epochs):
            epoch_start_time = time.time()
            total_pi_loss, total_z_values_loss, total_q_values_loss = 0.0, 0.0, 0.0
            logged_this_epoch = False

            epoch_grad_norms_before_clip = []

            for batch_states, batch_pis, batch_z_values, batch_q_values, moves_mask in self.train_loader:
                batch_states_bool = batch_states.to(self.device, non_blocking=non_block)
                batch_pis = batch_pis.to(self.device, non_blocking=non_block).float()
                batch_z_values = batch_z_values.to(self.device, non_blocking=non_block)
                batch_q_values = batch_q_values.to(self.device, non_blocking=non_block)
                moves_mask = moves_mask.to(self.device, non_blocking=non_block)

                self.model.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.scaler.is_enabled()):
                    pred_pi_logits, pred_z_values, pred_q_values = self.model.model(batch_states_bool.float())
                    masked_logits = pred_pi_logits.float().masked_fill(~moves_mask, -1e9)
                    loss_pi = F.cross_entropy(masked_logits, batch_pis)
                    loss_z_values = F.smooth_l1_loss(pred_z_values, batch_z_values)
                    loss_q_values = F.smooth_l1_loss(pred_q_values, batch_q_values, beta=0.5)
                    loss = pi_weight * loss_pi + values_z_weight * loss_z_values + values_q_weight * loss_q_values

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.model.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=2.5)
                self.scaler.step(self.model.optimizer)

                # Add to epoch grad norms
                epoch_grad_norms_before_clip.append(total_norm.float().item())

                # This should probably be generalized and not check for a specific model class
                # However, I am not sure if this even works / helps so...
                if isinstance(uncompiled_model, ThreePlayerChessformerBertV2):
                    with torch.no_grad():
                        clip_value = 4.50
                        uncompiled_model.relative_bias_table.weight.clamp_(-clip_value, clip_value)

                if isinstance(uncompiled_model, ThreePlayerChessformerShaw):
                    with torch.no_grad():
                        clip_value = 4.50
                        uncompiled_model.relative_pos_embedding_table.weight.clamp_(-clip_value, clip_value)

                if isinstance(uncompiled_model, ThreePlayerHybrid):
                    with torch.no_grad():
                        clip_value = 4.50
                        uncompiled_model.relative_pos_embedding_table.weight.clamp_(-clip_value, clip_value)

                self.scaler.update()

                if self.scheduler:
                    self.scheduler.step()

                total_pi_loss += loss_pi.float().item()
                total_z_values_loss += loss_z_values.float().item()
                total_q_values_loss += loss_q_values.float().item()

                if not logged_this_epoch:
                    log_message = "Random samples from the last batch:"
                    for i in range(4):
                        random_index = torch.randint(0, batch_states_bool.size(0), (1,)).item()
                        pred_z = pred_z_values[random_index].detach().float().cpu().numpy()
                        true_z = batch_z_values[random_index].detach().float().cpu().numpy()
                        pred_q = pred_q_values[random_index].detach().float().cpu().numpy()
                        true_q = batch_q_values[random_index].detach().float().cpu().numpy()
                        log_message += f"\nZ [predicted {pred_z} true {true_z}] Q [predicted {pred_q} true {true_q}]"
                    logging.info(log_message)
                    self.model.model.log_gradients(epoch)
                    for name, param in self.model.model.named_parameters():
                        if param.grad is not None:
                            self.summary_writer.add_histogram(f"grads/{name.replace('.', '/')}", param.grad.float(),
                                                              self.total_training_steps)
                    logged_this_epoch = True

                self.total_training_steps += 1

            duration_seconds = time.time() - epoch_start_time
            epoch_loss = (
                    pi_weight * total_pi_loss + values_z_weight * total_z_values_loss + values_q_weight * total_q_values_loss)
            avg_pi_loss = total_pi_loss / len(self.train_loader)
            avg_z_values_loss = total_z_values_loss / len(self.train_loader)
            avg_q_values_loss = total_q_values_loss / len(self.train_loader)
            avg_loss = epoch_loss / len(self.train_loader)

            avg_norm = sum(epoch_grad_norms_before_clip) / len(epoch_grad_norms_before_clip)

            with torch.no_grad():
                total_l2_reg_loss = 0.0
                total_weight_norm = 0.0
                # Iterate over optimizer's parameter groups to respect different weight_decay settings
                for group in self.model.optimizer.param_groups:
                    weight_decay = group['weight_decay']
                    for p in group['params']:
                        if p.requires_grad:
                            param_norm = p.data.norm(2)
                            param_norm_float = param_norm.float().item()
                            total_weight_norm += param_norm_float ** 2
                            if weight_decay > 0:
                                total_l2_reg_loss += weight_decay * (param_norm_float ** 2)
                total_weight_norm = total_weight_norm ** 0.5

            self.summary_writer.add_scalar("loss/l2_regularization", total_l2_reg_loss, self.total_training_steps)
            self.summary_writer.add_scalar("weights/l2_norm", total_weight_norm, self.total_training_steps)

            self.summary_writer.add_scalar("gradients/total_norm_before_clip", avg_norm, self.total_training_steps)
            self.summary_writer.add_scalar("loss/pi_loss", avg_pi_loss, self.total_training_steps)
            self.summary_writer.add_scalar("loss/z_values_loss", avg_z_values_loss, self.total_training_steps)
            self.summary_writer.add_scalar("loss/q_values_loss", avg_q_values_loss, self.total_training_steps)
            self.summary_writer.add_scalar("loss/total_weighted_loss", avg_loss, self.total_training_steps)
            self.summary_writer.add_scalar("learning_rate", self.model.optimizer.param_groups[0]['lr'],
                                           self.total_training_steps)

            logging.info(
                f"Epoch {epoch + 1}/{request.epochs} ({duration_seconds:.2f}s) | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Avg Pi Loss: {avg_pi_loss:.6f} | "
                f"Avg V_z Loss: {avg_z_values_loss:.6f} | "
                f"Avg V_q Loss: {avg_q_values_loss:.6f} | "
                f"LR: {self.model.optimizer.param_groups[0]['lr']:.2e} | "
                f"Training steps: {self.total_training_steps}."
            )

        self.model.version += 1
        model_path = self.model.save_checkpoint(
            pytorch_dir=self.settings.pytorch_dir,
            optimizer_dir=self.settings.optimizer_dir
        )
        onnx_path = self.model.save_to_onnx(
            onnx_dir=self.settings.onnx_dir,
            batch_size=self.settings.batch_size
        )
        duration_seconds = time.time() - start_time
        self.model.debug()
        logging.info(
            f"Model trained and saved at {model_path} and ONNX at {onnx_path} in {duration_seconds:.2f} seconds.")

        return training_pb2.TrainModelResponse(
            success=True,
            new_model_path=onnx_path.as_posix(),
            duration_seconds=duration_seconds,
            total_training_steps=self.total_training_steps,
        )

    def TensorboardScalar(self, request, context):
        if self.summary_writer is None:
            return settings_not_set_error()
        self.summary_writer.add_scalar(request.tag, request.value, global_step=request.step)
        return training_pb2.TensorboardResponse(success=True)

    def TensorboardMultipleScalars(self, request, context):
        if self.summary_writer is None:
            return settings_not_set_error()

        self.summary_writer.add_scalars(
            request.tag,
            request.values,
            global_step=request.step
        )

        return training_pb2.TensorboardResponse(success=True)

    def TensorboardHistogram(self, request, context):
        """
        Uses the RAW histogram API to approximate the histogram from bucket counts.
        """
        if self.summary_writer is None:
            return settings_not_set_error()
        bucket_counts = np.array(list(request.values), dtype=np.float32)
        num_buckets = len(bucket_counts)
        bucket_limits = np.arange(1, num_buckets + 1, dtype=np.float64)
        num = int(float(np.sum(bucket_counts)))
        non_zero_indices = np.where(bucket_counts > 0)[0]
        min_val = float(non_zero_indices[0])
        max_val = float(bucket_limits[non_zero_indices[-1]])
        bin_centers = bucket_limits - 0.5
        sum_val = float(np.sum(bucket_counts * bin_centers))
        sum_squares = float(np.sum(bucket_counts * (bin_centers ** 2)))
        self.summary_writer.add_histogram_raw(
            tag=request.tag, min=min_val, max=max_val, num=num, sum=sum_val,
            sum_squares=sum_squares, bucket_limits=bucket_limits.tolist(),
            bucket_counts=bucket_counts.tolist(), global_step=request.step
        )
        return training_pb2.TensorboardResponse(success=True)
