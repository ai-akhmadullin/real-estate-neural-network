public interface IActivationFunction {
    double Activate(double x);
    double Derivate(double x);
}

public class ReLUFunction : IActivationFunction {
    public double Activate(double x) {
        return Math.Max(0, x);
    }

    public double Derivate(double x) {
        return x > 0 ? 1 : 0;
    }
}

public class IdentityFunction : IActivationFunction {
    public double Activate(double x) {
        return x;
    }

    public double Derivate(double x) {
        return 1;
    }
}

