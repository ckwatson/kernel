import pytest
import numpy as np
from kernel.engine import handy_functions


def test_np_repr_and_np_print(capsys):
    arr = np.array([[1.123456789, 2], [3, 4]])
    # np_repr should return a string
    result = handy_functions.np_repr(arr)
    assert isinstance(result, str)
    # np_print should print the same string
    handy_functions.np_print(arr)
    captured = capsys.readouterr()
    assert result in captured.out


def test_file_error_and_warning(capsys):
    handy_functions.file_error('foo.txt', 'open')
    err = capsys.readouterr().err
    assert "ERROR could not open file: 'foo.txt'" in err
    handy_functions.warning('something went wrong')
    err = capsys.readouterr().err
    assert "WARNING:  something went wrong" in err


def test_negative_coefficient_exception():
    arr = [1, -2, 3]
    exc = handy_functions.NegativeCoefficientException(arr)
    assert exc.value == arr
    assert str(exc) == repr(arr)


def test_crash(monkeypatch):
    # Patch sys.exit to raise SystemExit so we can catch it
    monkeypatch.setattr('sys.exit', lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit):
        handy_functions.crash()
