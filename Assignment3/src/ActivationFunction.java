/*
 * Tanner Turba
 * December 3, 2024
 * CS 557 - Machine Learning
 * 
 * This is the enumeration that supports the extra credit activation function options. 
 * For each available option, the activation function, g, can be calculated as well as its derivative, g'.
 */
public enum ActivationFunction {
    LOGISTIC {
        @Override
        public double g(double inJ) {
            return 1.0 / (1.0 + Math.pow(Math.E, -inJ));
        }

        @Override
        public double gPrime(double inJ) {
            return g(inJ) * (1.0 - g(inJ));
        }
    },
    RELU {
        @Override
        public double g(double inJ) {
            if (inJ >= 0.0) {
                return inJ;
            }
            return 0.0;
        }

        @Override
        public double gPrime(double inJ) {
            if (inJ > 0) {
                return 1.0;
            }
            return 0.0;
        }
    },
    SOFTPLUS {
        @Override
        public double g(double inJ) {
            return Math.log(1 + Math.pow(Math.E, inJ));
        }

        @Override
        public double gPrime(double inJ) {
            return 1.0 / (1.0 + Math.pow(Math.E, -inJ));
        }
    },
    TANH {
        @Override
        public double g(double inJ) {
            return (Math.pow(Math.E, inJ) - Math.pow(Math.E, -inJ)) / (Math.pow(Math.E, inJ) + Math.pow(Math.E, -inJ));
        }

        @Override
        public double gPrime(double inJ) {
            return 1.0 - Math.pow(g(inJ), 2);
        }
    };

    // Activation function
    public abstract double g(double inJ);

    // Derivative of activation function
    public abstract double gPrime(double inJ);

    /**
     * Converts a string to its activation function.
     * @param func
     * @return
     */
    public static ActivationFunction getFunction(String func) {
        switch (func.toLowerCase()) {
            case "relu":
                return RELU;

            case "softplus":
                return SOFTPLUS;

            case "tanh":
                return TANH;
        
            default:
                return LOGISTIC;
        }
    }
}
