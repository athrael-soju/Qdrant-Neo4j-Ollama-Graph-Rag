import traceback

try:
    import graphrag
    print("Successfully imported graphrag")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc() 