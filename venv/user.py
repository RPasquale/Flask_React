from werkzeug.security import generate_password_hash, check_password_hash

class User:
    def __init__(self, id=None, full_name=None, email=None, password_hash=None, _id=None):
        self.id = id
        self.full_name = full_name
        self.email = email
        self.password_hash = password_hash
        self._id = _id

    def set_password(self, password):
        # Logic to hash and set the user's password
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        # Logic to check if the provided password matches the user's hashed password
        return check_password_hash(self.password_hash, password)



