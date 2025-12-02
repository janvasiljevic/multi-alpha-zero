use ndarray::{Array3, ArrayView, ArrayViewMut2, Ix3};
use ort::value::Tensor;

#[derive(Debug, Clone)]
pub struct Batch {
    array: Array3<bool>,
    batch_size: usize,
}

/// TransformerBatch always has a static batch size, since we use only ONNX networks with
/// a fixed batch size.
impl Batch {
    pub fn new(batch_size: usize, board_size: usize, field_size: usize) -> Self {
        let array = Array3::from_elem((batch_size, board_size, field_size), false);
        Batch { array, batch_size }
    }

    pub fn get_shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Get a view of the input tensor for the given index.
    /// Zero-copy, so no data is cloned.
    pub fn get_mut_item(&mut self, index: usize) -> ArrayViewMut2<'_, bool> {
        debug_assert!(
            index < self.batch_size,
            "Index {} out of bounds for batch size {}",
            index,
            self.batch_size
        );

        self.array.slice_mut(ndarray::s![index, .., ..])
    }

    pub fn view(&self) -> ArrayView<'_, bool, Ix3> {
        self.array.view()
    }

    pub fn tensor(&self) -> Tensor<bool> {
        Tensor::from_array(self.array.clone()).expect("Failed to convert array to Tensor")
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod mapper_tests {
    use super::*;
    use crate::mapping::hex_absolute_mapper::HexAbsoluteMapper;
    use crate::mapping::Board;
    use crate::mapping::InputMapper;
    use crate::mapping::PolicyMapper;
    use game_hex::game_hex::HexGame;

    #[test]
    // This needs to be ideally tested in a threaded context, with messages being sent to the GPU thread.
    fn test_batching_with_mapper() {
        let mut game = HexGame::new(3).unwrap(); // radius 1
        let mut move_store = Vec::new();
        let mapper = HexAbsoluteMapper::new(&game);
        let batch_size = 1024;
        let board_size = mapper.policy_len();
        let field_size = mapper.input_board_shape()[1];

        let mut batch = Batch::new(batch_size, board_size, field_size);

        game.fill_move_store(&mut move_store);

        for i in 0..batch_size {
            let mut input_view = batch.get_mut_item(i);
            mapper.encode_input(&mut input_view, &game);

            if game.is_terminal() {
                game = HexGame::new(3).unwrap(); // Reset the game if terminal
            }

            if let Some(mv) = game.legal_moves().get(i % game.legal_moves().len()) {
                game = game.play_move_clone(mv, &mut move_store);
            }
        }

        assert_eq!(batch.array.shape(), &[batch_size, board_size, field_size]);

        println!("Debug tensor: {:?}", batch.array);
    }
}
