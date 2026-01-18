import sqlite3
import os

def fix_database():
    # ⚠️ CHECK: Make sure this name matches your actual file (e.g., 'database.db')
    db_path = 'crop_doctor.db' 
    print(f"🔧 Connecting to database at: {os.path.abspath(db_path)}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ==========================================
    # 1. COMMUNITY POSTS TABLE
    # ==========================================
    print("📝 Checking/Creating 'community_posts' table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS community_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            crop_type TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            confidence TEXT NOT NULL,
            user_question TEXT NOT NULL,
            expert_reply TEXT,
            reply_author TEXT, 
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Double-check for the 'reply_author' column (for older databases)
    try:
        cursor.execute("ALTER TABLE community_posts ADD COLUMN reply_author TEXT")
        print("   ✅ Added missing column 'reply_author'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("   ✅ Column 'reply_author' already exists.")

    # ==========================================
    # 2. EXPERT TOKENS TABLE
    # ==========================================
    print("📝 Checking/Creating 'expert_tokens' table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expert_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            assigned_to_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ==========================================
    # 4. VERIFICATION REQUESTS TABLE (NEW)
    # ==========================================
    print("📝 Checking/Creating 'verification_requests' table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            qualification TEXT NOT NULL,
            role TEXT DEFAULT 'Expert',
            status TEXT DEFAULT 'pending',
            timestamp TEXT
        )
    ''')
    
    # Check for 'role' column (Update for existing databases)
    try:
        cursor.execute("ALTER TABLE verification_requests ADD COLUMN role TEXT DEFAULT 'Expert'")
        print("   ✅ Added missing column 'role' to verification_requests.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("   ✅ Column 'role' already exists in verification_requests.")
            
    print("📜 Creating/Updating 'government_schemes' table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS government_schemes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ministry TEXT,
            description TEXT,
            eligible_state TEXT DEFAULT 'All',
            eligible_crop TEXT DEFAULT 'All',
            benefit TEXT,
            link TEXT,
            deadline DATE
        )
    ''')
    
    # ✅ NEW ROBUST DATA (Verified HTTPS Links)
    schemes_data = [
        # --- ALL INDIA SCHEMES (Visible to everyone) ---
        ('Mission for Integrated Development of Horticulture (MIDH)', 'Ministry of Agriculture', 
         'Subsidy for planting material, greenhouses, and cold storage for fruit crops.', 
         'All', 'Apple, Grape, Cherry, Peach, Strawberry', '40-50% Subsidy on Infrastructure', 'https://midh.gov.in/', '2026-03-31'),
        
        ('Operation Greens (TOP Scheme)', 'Ministry of Food Processing', 
         'Price stabilization scheme specifically for Tomato, Onion, and Potato farmers.', 
         'All', 'Tomato, Potato', '50% Subsidy on Transport & Storage', 'https://www.mofpi.gov.in/', '2026-12-31'),
        
        ('Pradhan Mantri Fasal Bima Yojana (PMFBY)', 'Ministry of Agriculture', 
         'Insurance against crop loss due to pests (like Late Blight) or weather.', 
         'All', 'All', 'Insurance Claim Settlement', 'https://pmfby.gov.in/', '2026-07-31'),

        ('NFSM - Coarse Cereals (Maize)', 'Ministry of Agriculture', 
         'Support for improved seeds and technology demonstrations for Maize (Corn).', 
         'All', 'Corn', 'Free Hybrid Seeds & Field Demos', 'https://nfsm.gov.in/', '2026-06-30'),

        # --- NORTH INDIA SPECIFIC ---
        ('High Density Apple Plantation Scheme', 'Dept of Horticulture', 
         'Special subsidy for high-density apple orchards to boost production.', 
         'Jammu and Kashmir, Himachal Pradesh, Uttarakhand', 'Apple', '50% Subsidy on Plant Material', 'https://dirhortijmu.nic.in/', '2026-10-15'),

        ('Citrus Development Programme', 'National Horticulture Board', 
         'Rejuvenation of old orchards and new plantation support for Citrus fruits.', 
         'Punjab, Haryana, Rajasthan', 'Orange', 'Credit linked subsidy up to 40%', 'https://nhb.gov.in/', '2026-09-15'),

        # --- SOUTH INDIA SPECIFIC ---
        ('Drip Irrigation Subsidy (PMKSY)', 'Ministry of Jal Shakti', 
         'Massive subsidy for installing drip irrigation systems in water-scarce regions.', 
         'Tamil Nadu, Karnataka, Andhra Pradesh, Telangana', 'All', '75-100% Subsidy for Small Farmers', 'https://pmksy.gov.in/', '2026-05-20'),

        ('Coconut Development Board Scheme', 'Ministry of Agriculture', 
         'Assistance for expansion of area under coconut and replanting.', 
         'Kerala, Tamil Nadu, Karnataka, Goa', 'Coconut', '₹17,500 per hectare', 'https://www.coconutboard.gov.in/', '2026-08-10'),

        # --- CENTRAL & WEST INDIA ---
        ('NFSM - Oilseeds & Soybean', 'Ministry of Agriculture', 
         'Incentives for increasing oilseed production including Soybean.', 
         'Madhya Pradesh, Maharashtra, Gujarat', 'Soybean', '₹4000/quintal subsidy on seeds', 'https://nfsm.gov.in/', '2026-05-20'),

        ('Onion Storage Structure Scheme', 'Maharashtra Dept of Agriculture', 
         'Subsidy for constructing "Kanda Chawl" (Onion Storage) to prevent rotting.', 
         'Maharashtra, Gujarat', 'Onion', '₹87,500 per unit subsidy', 'https://krishi.maharashtra.gov.in/', '2026-04-01'),

        # --- EAST & NORTH-EAST INDIA ---
        ('BGREI - Rice & Veg Program', 'Ministry of Agriculture', 
         'Bringing Green Revolution to Eastern India - Tech support for rice and vegetables.', 
         'West Bengal, Bihar, Odisha, Assam', 'Rice, Tomato, Potato', 'Free Seeds & Tech Support', 'https://rkvy.nic.in/', '2026-06-15'),

        ('Organic Farming Mission (MOVCDNER)', 'Ministry of Agriculture', 
         'Promoting organic farming certification and value chains.', 
         'Sikkim, Assam, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Meghalaya', 'All', '₹50,000 per hectare for 3 years', 'https://pgsindia-ncof.gov.in/', '2026-11-30')
    ]

    # Clear old data to prevent mixing (This ensures you get the clean Horticulture list)
    cursor.execute("DELETE FROM government_schemes") 
    
    # Insert New Data
    cursor.executemany('''
        INSERT INTO government_schemes (name, ministry, description, eligible_state, eligible_crop, benefit, link, deadline)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', schemes_data)
    print("   ✅ Added RELEVANT Horticulture Schemes (Apple, Tomato, etc.).")

    conn.commit()
    conn.close()
    print("\n🚀 Database fixed! You can now run your app.")

if __name__ == "__main__":
    fix_database()