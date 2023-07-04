from pymongo import MongoClient
from user import User

uri = 'mongodb+srv://rpasquale01:Binga.12@mernpymlcluster.12c8kkg.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(uri)

# Create a new database for your cluster
# (database name = ApplicationUserDatabase)
db = client.ApplicationUserDatabase

# Add the User model to the database
db.users = User

def connect_to_database():
    client = MongoClient(uri)
    try:
        # Connect to the MongoDB Atlas database
        client.server_info()  # Check if connection is successful
        print("Connected to the database")

        # Return the database instance for further use
        database = client.get_database("ApplicationUserDatabase")
        print("Database Name:", database.name)

        return database
    except Exception as e:
        print("Error connecting to the database:", str(e))
        raise






