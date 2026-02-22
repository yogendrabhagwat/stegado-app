import sys
import traceback

try:
    from app import create_app
    app = create_app()
    print("✅ Stegado app created successfully!", flush=True)
except Exception as e:
    print(f"❌ FATAL: Failed to create app: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
