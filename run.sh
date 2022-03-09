 #!/bin/bash
sh run_CLASSLA.sh &
/opt/venv1/bin/uvicorn RECON.main:app --host 0.0.0.0 --port 8000