/// <summary>
/// Defines the contract for activation functions.
/// </summary>
public interface IActivationFunction {
    /// <summary>
    /// Applies the activation function to the specified input value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The activated value.</returns>
    double Activate(double x);

    /// <summary>
    /// Applies the derivative of the activation function to the specified input value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derived value.</returns>
    double Derivate(double x);
}

/// <summary>
/// Represents the Rectified Linear Unit (ReLU) activation function.
/// </summary>
public class ReLUFunction : IActivationFunction {
    /// <summary>
    /// Activates the input using the ReLU function: ReLU(x) = x if x > 0, 0 otherwise.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The activated value.</returns>
    public double Activate(double x) {
        return Math.Max(0, x);
    }

    /// <summary>
    /// Computes the derivative of the ReLU function: ReLU'(x) = 1 if x > 0, 0 otherwise.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derived value.</returns>
    public double Derivate(double x) {
        return x > 0 ? 1 : 0;
    }
}

/// <summary>
/// Represents the Identity activation function.
/// </summary>
public class IdentityFunction : IActivationFunction {
    /// <summary>
    /// Activates the input using the Identity function: f(x) = x.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The activated value.</returns>
    public double Activate(double x) {
        return x;
    }

    /// <summary>
    /// Computes the derivative of the Identity function: f'(x) = 1.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The derived value.</returns>
    public double Derivate(double x) {
        return 1;
    }
}

