# scripts/run_gunicorn.sh
#!/bin/bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 --access-logfile - --error-logfile - api.app:app