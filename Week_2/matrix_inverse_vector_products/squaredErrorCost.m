function cost = squaredErrorCost(A, b, x)
  cost = norm(A * x - b) ** 2
endfunction
