"""Initialize and serve the demo on Render's public HTTP port."""

import os

if not os.environ.get("EVOTING_ADMIN_PASSWORD"):
    raise RuntimeError("Set EVOTING_ADMIN_PASSWORD in the hosting environment")
if not os.environ.get("EVOTING_SECRET_KEY"):
    raise RuntimeError("Set EVOTING_SECRET_KEY in the hosting environment")

from waitress import serve

from app import app, initialize_project

initialize_project(download_models=False)

port = int(os.environ.get("PORT", "10000"))
print(f"Serving demo on 0.0.0.0:{port}")
serve(app, host="0.0.0.0", port=port, threads=4)
