"""
generate_data.py
Generates a synthetic fintech SQLite database for the FinAgent project.
Creates three tables: accounts, merchants, transactions.
No real Amex or personal data is used — everything here is fake/synthetic.
"""
 
import sqlite3
import random
from datetime import datetime, timedelta
from faker import Faker
 
fake = Faker()
Faker.seed(42)
random.seed(42)
 
DB_PATH = "finance.db"
 
CATEGORIES = [
    "groceries", "dining", "subscriptions", "travel", "utilities",
    "entertainment", "shopping", "healthcare", "transportation", "rent"
]
 
MERCHANT_NAMES_BY_CATEGORY = {
    "groceries": ["Whole Foods", "Trader Joe's", "Kroger", "Safeway"],
    "dining": ["Chipotle", "Local Bistro", "Starbucks", "Pizza Place"],
    "subscriptions": ["Netflix", "Spotify", "Amazon Prime", "Disney+"],
    "travel": ["Delta Airlines", "Marriott", "Uber", "Airbnb"],
    "utilities": ["ConEd", "Verizon", "PSEG", "Comcast"],
    "entertainment": ["AMC Theatres", "Ticketmaster", "Steam", "Apple Arcade"],
    "shopping": ["Amazon", "Target", "Best Buy", "Nike"],
    "healthcare": ["CVS Pharmacy", "Walgreens", "Quest Diagnostics"],
    "transportation": ["Shell Gas", "NJ Transit", "Lyft"],
    "rent": ["Property Management Co"]
}
 
 
def create_tables(conn):
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS transactions;
    DROP TABLE IF EXISTS merchants;
    DROP TABLE IF EXISTS accounts;
 
    CREATE TABLE accounts (
        account_id INTEGER PRIMARY KEY,
        account_holder TEXT NOT NULL,
        account_type TEXT NOT NULL,
        opened_date TEXT NOT NULL
    );
 
    CREATE TABLE merchants (
        merchant_id INTEGER PRIMARY KEY,
        merchant_name TEXT NOT NULL,
        category TEXT NOT NULL
    );
 
    CREATE TABLE transactions (
        transaction_id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL,
        merchant_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        transaction_date TEXT NOT NULL,
        category TEXT NOT NULL,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id),
        FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
    );
    """)
    conn.commit()
 
 
def generate_accounts(conn, n=5):
    cur = conn.cursor()
    account_types = ["checking", "savings", "credit"]
    for i in range(1, n + 1):
        cur.execute(
            "INSERT INTO accounts (account_id, account_holder, account_type, opened_date) VALUES (?, ?, ?, ?)",
            (i, fake.name(), random.choice(account_types), fake.date_between(start_date="-5y", end_date="-1y").isoformat())
        )
    conn.commit()
    return list(range(1, n + 1))
 
 
def generate_merchants(conn):
    cur = conn.cursor()
    merchant_id = 1
    merchant_map = {}
    for category, names in MERCHANT_NAMES_BY_CATEGORY.items():
        for name in names:
            cur.execute(
                "INSERT INTO merchants (merchant_id, merchant_name, category) VALUES (?, ?, ?)",
                (merchant_id, name, category)
            )
            merchant_map[merchant_id] = category
            merchant_id += 1
    conn.commit()
    return merchant_map
 
 
def generate_transactions(conn, account_ids, merchant_map, n=500):
    cur = conn.cursor()
    merchant_ids = list(merchant_map.keys())
 
    # Rough amount ranges by category for realism
    amount_ranges = {
        "groceries": (20, 150),
        "dining": (10, 80),
        "subscriptions": (5, 20),
        "travel": (100, 800),
        "utilities": (40, 200),
        "entertainment": (10, 60),
        "shopping": (15, 300),
        "healthcare": (10, 250),
        "transportation": (5, 90),
        "rent": (1200, 2200),
    }
 
    today = datetime.now()
    for tx_id in range(1, n + 1):
        account_id = random.choice(account_ids)
        merchant_id = random.choice(merchant_ids)
        category = merchant_map[merchant_id]
        low, high = amount_ranges[category]
        amount = round(random.uniform(low, high), 2)
        days_ago = random.randint(0, 180)  # last ~6 months
        tx_date = (today - timedelta(days=days_ago)).date().isoformat()
 
        cur.execute(
            "INSERT INTO transactions (transaction_id, account_id, merchant_id, amount, transaction_date, category) VALUES (?, ?, ?, ?, ?, ?)",
            (tx_id, account_id, merchant_id, amount, tx_date, category)
        )
    conn.commit()
 
 
def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    account_ids = generate_accounts(conn, n=5)
    merchant_map = generate_merchants(conn)
    generate_transactions(conn, account_ids, merchant_map, n=500)
    conn.close()
    print(f"Done. Database created at: {DB_PATH}")
    print(f"   - {len(account_ids)} accounts")
    print(f"   - {len(merchant_map)} merchants")
    print(f"   - 500 transactions")
 
 
if __name__ == "__main__":
    main()