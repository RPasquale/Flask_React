const { MongoClient } = require("mongodb");

const uri =
  "mongodb+srv://rpasquale01:Binga.12@mernpymlcluster.12c8kkg.mongodb.net/?retryWrites=true&w=majority";

async function connectToDatabase() {
  const client = new MongoClient(uri, { useUnifiedTopology: true });
  try {
    // Connect to the MongoDB Atlas database
    await client.connect();
    console.log("Connected to the database");

    // Return the database instance for further use
    return client.db();
  } catch (error) {
    console.error("Error connecting to the database:", error);
    throw error;
  }
}

module.exports = connectToDatabase;
