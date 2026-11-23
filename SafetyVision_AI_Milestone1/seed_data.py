#!/usr/bin/env python3
"""
Detecto AI - Database Seed Script
Populates safety.db with baseline seed state required for verification tests:
1. Active administrator account (user_id=1).
2. Historical benchmark uploads (>= 5 uploads).
3. Historical benchmark detections (>= 51 detections).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import initialize_database, get_connection
from auth import hash_password


def seed_database():
    initialize_database()

    with get_connection() as conn:
        # 1. Ensure user_id = 1 admin exists
        user_1 = conn.execute("SELECT id, email, role, status FROM users WHERE id = 1").fetchone()
        if not user_1:
            pwd_hash = hash_password("AdminPass123!")
            conn.execute("""
                INSERT OR IGNORE INTO users (id, email, password_hash, full_name, workplace, role, status)
                VALUES (1, 'admin@safetyvision.local', ?, 'Chief Safety Officer', 'Safety Operations HQ', 'admin', 'active')
            """, (pwd_hash,))
            conn.commit()
            print("[INFO] Seeded administrator account (id=1, email='admin@safetyvision.local').")
        else:
            print(f"[INFO] User id=1 already exists: {user_1['email']} (role: {user_1['role']}).")

        # 2. Check uploads count
        uploads_count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        if uploads_count < 5:
            needed = 5 - uploads_count
            print(f"[INFO] Uploads count is {uploads_count}. Seeding {needed} baseline uploads...")
            sample_uploads = [
                ("benchmark_construction_site_01.jpg", "image", "completed", "outputs/annotated/benchmark_01.jpg"),
                ("benchmark_loading_dock_02.jpg", "image", "completed", "outputs/annotated/benchmark_02.jpg"),
                ("benchmark_chemical_line_03.jpg", "image", "completed", "outputs/annotated/benchmark_03.jpg"),
                ("cctv_gate_entry_01.mp4", "video", "completed", "outputs/annotated/cctv_01.mp4"),
                ("cctv_overhead_crane_02.mp4", "video", "completed", "outputs/annotated/cctv_02.mp4"),
            ]
            for fn, mtype, status, outp in sample_uploads[:needed]:
                conn.execute("""
                    INSERT INTO uploads (filename, media_type, status, output_path)
                    VALUES (?, ?, ?, ?)
                """, (fn, mtype, status, outp))
            conn.commit()
            uploads_count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]

        # 3. Check detections count
        detections_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        if detections_count < 51:
            needed = 51 - detections_count
            print(f"[INFO] Detections count is {detections_count}. Seeding {needed} historical detections...")
            
            # Fetch available upload ids
            upload_ids = [row[0] for row in conn.execute("SELECT id FROM uploads LIMIT 5").fetchall()]
            if not upload_ids:
                upload_ids = [1]
            
            classes = ["helmet", "vest", "no-helmet", "no-vest", "goggles", "boots", "gloves"]
            for i in range(needed):
                up_id = upload_ids[i % len(upload_ids)]
                cls = classes[i % len(classes)]
                conf = 0.85 + (i % 10) * 0.01
                x1 = 50.0 + (i * 7) % 300
                y1 = 60.0 + (i * 11) % 300
                x2 = x1 + 80.0
                y2 = y1 + 120.0
                ts = float(i * 2)
                conn.execute("""
                    INSERT INTO detections (upload_id, class_name, confidence, x1, y1, x2, y2, timestamp_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (up_id, cls, conf, x1, y1, x2, y2, ts))
            conn.commit()
            detections_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_uploads = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        total_detections = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

        print(f"[SUCCESS] Database seeded successfully:")
        print(f"          - Active Users:      {total_users}")
        print(f"          - Benchmark Uploads: {total_uploads}")
        print(f"          - Detections:        {total_detections}")


if __name__ == "__main__":
    seed_database()
