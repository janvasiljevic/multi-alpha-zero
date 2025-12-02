import {MoveType, MoveWrapper, Piece} from "tri-hex-chess";
import {getX, getY, rotate} from "../lib/utils/utils_xy";
import {SIZE} from "../lib/constants";

export class NewMove {
    ref: MoveWrapper;
    x: number;
    y: number;
    color: string;
    rotation: number;
    notationLan: string;

    constructor(ref: MoveWrapper, rotation: number) {
        this.ref = ref;

        this.rotation = rotation;
        this.color = this._setColor();
        const rot = rotate(getX(this.ref.to.q, SIZE), getY(this.ref.to.q, this.ref.to.r, SIZE), rotation);
        this.x = rot.x;
        this.y = rot.y;
        this.notationLan = ref.getNotationLAN;
    }

    _setColor = () => {
        switch (this.ref.move_type) {
            case MoveType.Move:
            case MoveType.DoublePawnPush:
                return "orange";
            case MoveType.Capture:
            case MoveType.EnPassant:
            case MoveType.EnPassantPromotion:
                return "red";
            case MoveType.CastleKingSide:
            case MoveType.CastleQueenSide:
                return "blue";
            case MoveType.Promotion:
            case MoveType.CapturePromotion:
                return "green";
        }
    };

    getPromotionSquares() {
        if (!this.isPromotion()) return [];

        const proms = [Piece.Queen, Piece.Rook, Piece.Bishop, Piece.Knight];
        const proms_texts = ["Q", "R", "B", "N"];
        const {q, r} = this.ref.to;
        const offsets = [
            [0, 0],
            [-1, 0],
            [1, -1],
            [0, -1],
        ];

        return offsets.map(([dq, dr], i) => ({
            type: proms[i],
            text: proms_texts[i],
            ...rotate(getX(q + dq, SIZE), getY(q + dq, r + dr, SIZE), this.rotation),
        }));
    }

    isPromotion() {
        return (
            this.ref.move_type === MoveType.Promotion ||
            this.ref.move_type === MoveType.CapturePromotion ||
            this.ref.move_type === MoveType.EnPassantPromotion
        );
    }

    isCapture() {
        return this.ref.move_type === MoveType.Capture || this.ref.move_type === MoveType.CapturePromotion;
    }
}
