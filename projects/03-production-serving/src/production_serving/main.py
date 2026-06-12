import uvicorn

from production_serving.app import create_app
from production_serving.config import serving_host, serving_port


def main() -> None:
    uvicorn.run(create_app(), host=serving_host(), port=serving_port())


if __name__ == "__main__":  # pragma: no cover
    main()
