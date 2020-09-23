# Next, let’s create a file that will serve as the entry point for our application
# This will tell our Gunicorn server how to interact with the app

from flaskunetserver import app

if __name__ == "__main__":
    app.run()
