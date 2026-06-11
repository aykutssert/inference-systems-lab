import uvicorn

from local_inference.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
