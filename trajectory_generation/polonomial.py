import numpy as np

class Polynomial:
    def __init__(self, polynomial: np.ndarray) -> None:
        self.polynomial = polynomial
        self.number_of_terms = np.shape(polynomial)[0]

    def differantiate_polynomial(self, order_of_derivative: int) -> np.ndarray:
        derivative = np.zeros(self.number_of_terms)
        if self.number_of_terms - order_of_derivative >= 0:
            for i in range(order_of_derivative, self.number_of_terms, 1):
                new_order, coefficient = differantiate_monomial(i, order_of_derivative)
                derivative[new_order] = coefficient*self.polynomial[i]
        return derivative

    def sample_polynomial(self, x: int) -> float:
        '''
        given polynomial and x, calculate sum(polynomial[i]*x^i)
        '''
        exponent = np.array(range(self.number_of_terms))
        result = self.polynomial @ (x**exponent)
        return result

def differantiate_monomial(order_of_monomial: int, order_of_derivative: int) -> tuple[int, int]:
    '''
    calculate i^th order of t^n
    Args:
        order_of_monomial: n
        order_of_derivative: number of attempts to differantiate
    Returns:
        differantiated_order: order of monomial after differantiation
        coefficient: coefficient generated from differantiation
    '''
    differantiated_order = 0
    coefficient = 0
    if order_of_derivative <= order_of_monomial:
        coefficient = np.prod(range(order_of_monomial, order_of_monomial - order_of_derivative, -1))
        differantiated_order = order_of_monomial - order_of_derivative
    return (differantiated_order, coefficient)


if __name__ == "__main__":

    # x^3 2th derivative to be 6x
    diff_order, coeff = differantiate_monomial(3, 2)
    print(diff_order, coeff)
    # x^3 3th derivative to be 6
    diff_order, coeff = differantiate_monomial(3, 3)
    print(diff_order, coeff)
    
    polynomial_instance = Polynomial(np.array([3, 1, 1, 2]))    
    coeff = polynomial_instance.differantiate_polynomial(2)
    print(coeff)
