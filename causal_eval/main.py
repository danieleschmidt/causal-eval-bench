"""Main application entry point."""

import uvicorn
from causal_eval.api.app import create_app


def main():
    """Run the application."""
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()