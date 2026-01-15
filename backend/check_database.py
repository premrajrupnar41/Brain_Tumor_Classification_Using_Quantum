#!/usr/bin/env python3
"""
Check database contents directly without using DB Browser
"""
import sqlite3
import os

# Path to database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app", "auth.db")

print("=" * 60)
print("DATABASE CHECKER TOOL")
print("=" * 60)
print(f"\nDatabase path: {DB_PATH}")
print(f"Database exists: {os.path.exists(DB_PATH)}")

if not os.path.exists(DB_PATH):
    print("✗ Database file not found!")
    exit(1)

try:
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\n✓ Database connected successfully")
    print(f"Tables found: {len(tables)}")
    
    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"Columns: {[col[1] for col in columns]}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"Total rows: {row_count}")
        
        # Get all data
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            
            for i, row in enumerate(rows, 1):
                print(f"\nRow {i}:")
                for col, value in zip([col[1] for col in columns], row):
                    # Hide passwords for security
                    if col == "password":
                        display_value = value[:20] + "..." if len(value) > 20 else value
                    else:
                        display_value = value
                    print(f"  {col}: {display_value}")
        else:
            print("  (No data in this table)")
    
    conn.close()
    print("\n" + "=" * 60)
    print("✓ Database check completed successfully!")
    print("=" * 60)

except sqlite3.Error as e:
    print(f"\n✗ Database error: {e}")
except Exception as e:
    print(f"\n✗ Error: {e}")
