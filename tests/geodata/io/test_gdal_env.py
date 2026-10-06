import os

from geosave_engine.geodata.io.gdal_env import configure_gdal


def test_gdal_configuration_sets_supplied_values(monkeypatch) -> None:
    monkeypatch.delenv("GDAL_HTTP_MAX_RETRY", raising=False)
    monkeypatch.delenv("AWS_NO_SIGN_REQUEST", raising=False)

    configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"
    assert os.environ["GDAL_HTTP_MAX_RETRY"] == "3"


def test_gdal_configuration_points_s3_at_another_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_VIRTUAL_HOSTING", raising=False)

    configure_gdal(aws_s3_endpoint="s3.hf.co", aws_virtual_hosting=False)

    assert os.environ["AWS_S3_ENDPOINT"] == "s3.hf.co"
    assert os.environ["AWS_VIRTUAL_HOSTING"] == "FALSE"
