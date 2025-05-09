#Q1
import numpy as np
import matplotlib.pyplot as plt
# define x values from -10 to 10
x = np.arange(-10, 10, 0.1)
# define quadratic function y = x^2 + 4x + 10
y = x**2 + 4*x + 10
# print the minimum value of y
print("Minimum value of y:", np.min(y))
# plot the function
plt.plot(x, y)
plt.title("y = x^2 + 4x + 10")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

#Q2:
from deck import Deck, Card

class Hand:
    def __init__(self, deck):
        # draw 5 cards from deck
        cards = []
        for i in range(5):
            cards.append(deck.deal())
        self._cards = cards

    @property
    def cards(self):
        return self._cards

    def __str__(self):
        return str(self.cards)

    @property
    def is_flush(self):
        for card in self.cards[1:]:
            if self.cards[0].suit != card.suit:
                return False
        return True

    @property
    def num_matches(self):
        matches = 0
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                if self.cards[i].rank == self.cards[j].rank:
                    matches += 1
        return matches

    @property
    def is_pair(self):
        return self.num_matches == 2

    @property
    def is_2_pair(self):
        return self.num_matches == 4

    @property
    def is_trips(self):
        return self.num_matches == 6  # 3 of a kind

    @property
    def is_quads(self):
        return self.num_matches == 12  # 4 of a kind

    @property
    def is_full_house(self):
        return self.num_matches == 8  # 3 + 2

    @property
    def is_straight(self):
        if self.num_matches != 0:
            return False
        self.cards.sort()
        if Card.RANKS.index(self.cards[-1].rank) != Card.RANKS.index(self.cards[0].rank) + 4:
            return False
        return True

# simulate probability of Three of a Kind
matches = 0
count = 0
while matches < 1000:
    deck = Deck()
    deck.shuffle()
    hand = Hand(deck)
    count += 1
    if hand.is_trips:  # ← changed from is_straight
        matches += 1

print(f"The probability of Three of a Kind is {100 * matches / count}%")


#Q3
import numpy as np

a = np.arange(0, 12)
a = a.reshape(2, 6)

# transform the first row using a custom formula
a[0] = a[0]**2 + 2

# transform the second row using another formula
a[1] = a[1] * 4 + 4

print(a)

#Q4
def __str__(self):
    return f"{self._rank}{self._suit}"

def __eq__(self, other):
    return self.rank == other.rank

#Q5
import pandas as pd
import requests
import matplotlib.pyplot as plt

# download CSV file
ticker = "F"
url = f"https://raw.githubusercontent.com/itb-ie/midterm_data/refs/heads/main/{ticker}.csv"
with open("company.csv", "w") as f:
    f.write(requests.get(url).text)

# load CSV and parse date index
df = pd.read_csv("company.csv", index_col="Date")
df.index = pd.to_datetime(df.index)  # needed to filter by month/day

# 1. how many rows?
print("Number of rows:", len(df))

# 2. what are the column names?
print("Column names:", df.columns)

# 3. opening stock value on April 9 (any year)
april_9_open = df[(df.index.month == 4) & (df.index.day == 9)]["Open"]
print("Opening stock value(s) on April 9:")
print(april_9_open)

# 4. plot a column (example: 'Close')
df["Close"].plot(title="Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

#Q6
class Point:
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

class ColorPoint(Point):
    def __init__(self, x=0, y=0, color="red"):
        super().__init__(x, y)
        self._color = color

#Q8
    class Point:
        def __init__(self, x=0, y=0):
            self._x = x
            self._y = y

        @classmethod
        def from_string(cls, s):
            x_str, y_str = s.split(",")
            return cls(int(x_str), int(y_str))

#Q9
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(4, 4), index=[1, 2, 3, 4], columns=['a', 'b', 'c', 'd'])
print(df)

# Choose row with label 2 and column 'b'
# 1. With .loc (label-based)
print("Using loc:", df.loc[2, 'b'])

# 2. With .iloc (position-based → row 1, col 1)
print("Using iloc:", df.iloc[1, 1])

# 3. With .at (fast label access, index-style)
print("Using .at:", df.at[2, 'b'])

