import pytest
from calculator import divide

def test_divide():
    assert divide(10, 2) == 5
    assert divide(-10, 2) == -5
    assert divide(10, -2) == -5
    assert divide(0, 1) == 0
    with pytest.raises(ValueError, match='Cannot divide by zero'):
        divide(10, 0)