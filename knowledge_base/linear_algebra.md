# Linear Algebra – Basics

## Matrix Addition
Only possible when matrices have same order.

## Matrix Multiplication
A(m×n) · B(n×p) ⇒ result is (m×p)

## Determinant of 2×2
|a b|
|c d| = ad - bc

## Inverse of 2×2 Matrix
A⁻¹ = (1/det A) ·
| d  -b |
| -c  a |

Condition: det A ≠ 0

## System of Linear Equations
Unique solution if det ≠ 0  
No solution or infinite solutions if det = 0

## Common Mistakes
- Multiplying in wrong order
- Ignoring determinant zero case

## Determinants - Quick Checks
VALUE = 0 IF:
- Two rows/columns proportional
- Two rows/columns equal
OPERATIONS:
- Swap rows → sign changes
- Multiply row by k → determinant multiplies by k
- Add multiple of one row to another → no change

## Matrices - Key Rules
INVERSE EXISTS IF:
- det(A) ≠ 0
2x2 INVERSE:
|a b|        (1/det) | d −b|
|c d| → inverse =          |−c  a|

## System of Linear Equations
METHODS:
- Use determinants for 2–3 variables
- Rank method for consistency
CONDITIONS:
- Unique solution → rank(A) = rank(A|B) = number of variables
