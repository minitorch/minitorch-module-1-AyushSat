from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    list_vals_minus_ep, list_vals_plus_ep = [val for val in vals], [val for val in vals]
    if arg >= 0 and arg < len(list_vals_minus_ep):
        list_vals_minus_ep[arg] = list_vals_minus_ep[arg] - epsilon
        list_vals_plus_ep[arg] = list_vals_plus_ep[arg] + epsilon
    return (f(*list_vals_plus_ep) - f(*list_vals_minus_ep)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    top_sort: list[Variable] = []
    marked: set[int] = set()

    def visit(node: Variable) -> None:
        if node.unique_id in marked:
            return

        for p in node.parents:
            visit(p)
        marked.add(node.unique_id)
        top_sort.append(node)

    visit(variable)
    return top_sort


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    top_sort: list[Variable] = list(topological_sort(variable))[::-1]
    intermediate_mapping: dict[int, float] = {} # map of scalar unique id's to current derivative

    def sub(var: Variable, init_deriv: Any = None) -> None:
        if var.is_leaf():
            if var.unique_id in intermediate_mapping:
                var.accumulate_derivative(intermediate_mapping[var.unique_id])
            return

        curr_deriv = intermediate_mapping.get(var.unique_id, init_deriv)
        res: Iterable[Tuple[Variable, Any]] = var.chain_rule(curr_deriv)
        for chainvar, chainval in res:
            if chainvar.unique_id in intermediate_mapping:
                intermediate_mapping[chainvar.unique_id] += chainval
            else:
                intermediate_mapping[chainvar.unique_id] = chainval

    for v in top_sort:
        sub(v, init_deriv=deriv)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
