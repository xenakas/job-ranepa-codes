import pytest
from test_init import files, parse_texts  # , ids


@pytest.mark.parametrize('f', files)
def test_docx(f):
    # f_id = f[0]
    output = parse_texts(f)
    # it is a string
    if len(output['characteristics_pandas']) < 5:
        print(f)
        print(output)
    assert len(output['characteristics_pandas']) >= 5
    # assert len(ids[f_id]) <= len(output['characteristics_pandas'])
