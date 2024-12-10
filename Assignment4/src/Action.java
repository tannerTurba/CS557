public enum Action {
    UP, LEFT, RIGHT, DOWN;

    public String toString() {
        switch (this) {
            case UP:
                return "^";

            case DOWN:
                return "v";

            case LEFT:
                return "<";

            case RIGHT:
                return ">";

            default:
                return "";
        }
    }
}
