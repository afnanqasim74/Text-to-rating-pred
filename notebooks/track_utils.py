# Load Database Pkg
import sqlite3
#conn = sqlite3.connect('data.db')
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

"""
# Fxn
def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)')

def add_page_visited_details(pagename,timeOfvisit):
	c.execute('INSERT INTO pageTrackTable(pagename,timeOfvisit) VALUES(?,?)',(pagename,timeOfvisit))
	conn.commit()

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable')
	data = c.fetchall()
	return data
"""
# Fxn To Track Input & Prediction
def create_rating_table():
	c.execute('''CREATE TABLE IF NOT EXISTS rating(text TEXT,rat INTEGER,probability INTEGER)''')

def add_rating_details(t,r,p):
    c.execute("INSERT INTO rating (text, rat, probability) VALUES (?,?,?)", (t, r, p))
  

def view_all_prediction_details():
	c.execute('SELECT * FROM rating')
	data = c.fetchall()
	return data