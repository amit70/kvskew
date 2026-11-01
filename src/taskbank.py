"""
taskbank.py — 120 ground-truth tasks, stratified by how much the answer
depends on the prompt's representation.

STRATA
------
recall    : single-fact lookup. The question alone overdetermines the answer,
            so a contaminated prefix representation should matter least.
arith     : multi-step arithmetic. The answer is determined but requires
            carrying intermediate state through the generation.
context   : the answer depends on information supplied IN the prompt. This is
            the stratum where prefix contamination should hurt most, because
            the needed facts live in the cached tokens themselves.

MECHANISM PREDICTION (stated before running)
--------------------------------------------
break rate:  context > arith > recall

If the break rate is flat across strata, the effect is generic perturbation
noise rather than prefix contamination, which weakens the mechanism story even
if the aggregate rate is non-zero.

Each entry: (stratum, prompt, [accepted answer substrings])
Answers are lower-cased substring matches, so accepted lists include digit and
word forms where both are plausible.
"""

RECALL = [
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the capital of Brazil?", ["brasilia", "brasília"]),
    ("What is the capital of Egypt?", ["cairo"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
    ("Which ocean is the largest?", ["pacific"]),
    ("What is the longest river in the world?", ["nile", "amazon"]),
    ("What is the chemical symbol for water?", ["h2o", "h₂o"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What gas do humans breathe in to survive?", ["oxygen"]),
    ("How many days are in a week?", ["7", "seven"]),
    ("How many hours are in a day?", ["24", "twenty-four", "twenty four"]),
    ("How many sides does a triangle have?", ["3", "three"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("How many continents are there?", ["7", "seven"]),
    ("What is the opposite of 'hot'?", ["cold"]),
    ("What is the opposite of 'ancient'?", ["modern", "new", "recent"]),
    ("What color do you get mixing red and white?", ["pink"]),
    ("What color do you get mixing blue and yellow?", ["green"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
    ("What is the boiling point of water in Celsius?", ["100", "hundred"]),
    ("In which continent is Egypt located?", ["africa"]),
    ("In which continent is Peru located?", ["south america"]),
    ("Who wrote the play 'Romeo and Juliet'?", ["shakespeare"]),
    ("What is the currency of Japan?", ["yen"]),
    ("What is the currency of the United Kingdom?", ["pound", "sterling"]),
    ("How many legs does a spider have?", ["8", "eight"]),
    ("How many players are on a soccer team on the field?", ["11", "eleven"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What is the square root of 64?", ["8", "eight"]),
    ("What is the largest mammal?", ["blue whale", "whale"]),
    ("What language is primarily spoken in Brazil?", ["portuguese"]),
    ("How many minutes are in an hour?", ["60", "sixty"]),
    ("What season comes after summer?", ["autumn", "fall"]),
    ("What is the hardest natural substance?", ["diamond"]),
    ("How many strings does a standard guitar have?", ["6", "six"]),
    ("What is the main gas in Earth's atmosphere?", ["nitrogen"]),
    ("What is the capital of France?", ["paris"]),
]

ARITH = [
    ("What is 12 plus 15?", ["27"]),
    ("What is 9 times 6?", ["54"]),
    ("What is 100 divided by 4?", ["25"]),
    ("What is 17 plus 28?", ["45"]),
    ("What is 63 minus 19?", ["44"]),
    ("What is 8 times 7?", ["56"]),
    ("What is 144 divided by 12?", ["12"]),
    ("What is 25 times 4?", ["100"]),
    ("What is 90 minus 37?", ["53"]),
    ("What is 13 plus 29?", ["42"]),
    ("What is 7 times 12?", ["84"]),
    ("What is 81 divided by 9?", ["9"]),
    ("What is 45 plus 55?", ["100"]),
    ("What is 200 minus 76?", ["124"]),
    ("What is 15 times 3?", ["45"]),
    ("What is 96 divided by 8?", ["12"]),
    ("What is 34 plus 48?", ["82"]),
    ("What is 11 times 11?", ["121"]),
    ("What is 150 minus 65?", ["85"]),
    ("What is 6 times 15?", ["90"]),
    ("If you buy 3 items at 12 dollars each, what is the total?", ["36"]),
    ("If a train travels 60 miles per hour for 2 hours, how far does it go?",
     ["120"]),
    ("A box holds 8 pencils. How many pencils are in 7 boxes?", ["56"]),
    ("You have 50 dollars and spend 18. How much is left?", ["32"]),
    ("A shirt costs 20 dollars with a 5 dollar discount. What is the price?",
     ["15"]),
    ("If 4 friends split 48 dollars equally, how much does each get?", ["12"]),
    ("A recipe needs 3 eggs per cake. How many eggs for 5 cakes?", ["15"]),
    ("A book has 200 pages. You read 75. How many remain?", ["125"]),
    ("There are 24 students in 3 equal groups. How many per group?", ["8"]),
    ("A car uses 5 gallons per 100 miles. How many gallons for 300 miles?",
     ["15"]),
    ("What is 19 plus 19?", ["38"]),
    ("What is 5 times 24?", ["120"]),
    ("What is 300 divided by 6?", ["50"]),
    ("What is 88 minus 39?", ["49"]),
    ("What is 14 times 5?", ["70"]),
    ("What is 72 divided by 6?", ["12"]),
    ("What is 125 plus 75?", ["200"]),
    ("What is 16 times 4?", ["64"]),
    ("What is 500 minus 250?", ["250"]),
    ("What is 21 times 3?", ["63"]),
]

# context: the answer is ONLY derivable from facts stated in the prompt.
CONTEXT = [
    ("Maria has 3 cats and 2 dogs. Tom has 4 cats. How many cats in total?",
     ["7", "seven"]),
    ("The blue box weighs 5 kg. The red box weighs 8 kg. What is the total "
     "weight?", ["13"]),
    ("Sam is 12 years old. His sister is 5 years older. How old is his "
     "sister?", ["17", "seventeen"]),
    ("The store opens at 9 and closes at 17. How many hours is it open?",
     ["8", "eight"]),
    ("A shelf holds 6 books. There are 4 shelves. How many books?",
     ["24", "twenty-four"]),
    ("Lena ran 3 km on Monday and 5 km on Tuesday. How far in total?",
     ["8", "eight"]),
    ("The recipe serves 4. You double it. How many does it serve?",
     ["8", "eight"]),
    ("A bag has 10 red and 6 blue marbles. How many marbles total?",
     ["16", "sixteen"]),
    ("Zara earns 15 dollars an hour and works 4 hours. What does she earn?",
     ["60", "sixty"]),
    ("The flight leaves at 14:00 and lasts 3 hours. When does it land?",
     ["17", "5 pm", "5pm"]),
    ("There were 20 apples. 7 were eaten. How many are left?",
     ["13", "thirteen"]),
    ("A class has 15 boys and 13 girls. How many students?",
     ["28", "twenty-eight"]),
    ("The temperature was 12 degrees and rose by 9. What is it now?",
     ["21", "twenty-one"]),
    ("A tank holds 40 litres and is half full. How many litres are in it?",
     ["20", "twenty"]),
    ("Ben has 25 dollars. He spends 9 on lunch. How much remains?",
     ["16", "sixteen"]),
    ("The film starts at 19:00 and runs 2 hours. When does it end?",
     ["21", "9 pm", "9pm"]),
    ("A farm has 8 cows and triple that many sheep. How many sheep?",
     ["24", "twenty-four"]),
    ("Priya read 40 pages, then 35 more. How many pages has she read?",
     ["75", "seventy-five"]),
    ("A team scored 3 goals in the first half and 2 in the second. Total?",
     ["5", "five"]),
    ("The box contains 4 rows of 7 chocolates. How many chocolates?",
     ["28", "twenty-eight"]),
    ("Anna's car holds 45 litres. She adds 20 litres to an empty tank. "
     "How many more litres fit?", ["25", "twenty-five"]),
    ("A room is 5 metres long and 4 metres wide. What is its area?",
     ["20", "twenty"]),
    ("Jon saved 30 dollars in May and 45 in June. How much did he save?",
     ["75", "seventy-five"]),
    ("There are 9 chairs per table and 5 tables. How many chairs?",
     ["45", "forty-five"]),
    ("A jug holds 2 litres. How many millilitres is that?", ["2000", "2,000"]),
    ("Kim walked 1200 metres. How many kilometres is that?", ["1.2"]),
    ("A shop sold 18 shirts on Friday and 22 on Saturday. How many total?",
     ["40", "forty"]),
    ("The bus fare is 3 dollars each way. What is the round trip cost?",
     ["6", "six"]),
    ("A pizza is cut into 8 slices and 3 are eaten. How many remain?",
     ["5", "five"]),
    ("Omar has twice as many pens as Ada. Ada has 7 pens. How many has Omar?",
     ["14", "fourteen"]),
    ("A phone costs 300 dollars with 10 percent off. What is the discount?",
     ["30", "thirty"]),
    ("The library lends books for 21 days. You have had it 14 days. "
     "How many days remain?", ["7", "seven"]),
    ("A bottle holds 500 ml. How many bottles fill 2 litres?",
     ["4", "four"]),
    ("Ravi's train leaves in 45 minutes at 10:15. What time is it now?",
     ["9:30", "09:30", "half past nine"]),
    ("A garden has 6 rows with 9 plants each. How many plants?",
     ["54", "fifty-four"]),
    ("Two boxes weigh 7 kg each. What is the combined weight?",
     ["14", "fourteen"]),
    ("A worker earns 120 dollars for 8 hours. What is the hourly rate?",
     ["15", "fifteen"]),
    ("The tank loses 3 litres a day. How much in a week?",
     ["21", "twenty-one"]),
    ("A recipe uses 250 g of flour. How much for 4 batches?",
     ["1000", "1 kg", "1kg"]),
    ("Mia had 100 stickers, gave away 35, then got 10 more. How many now?",
     ["75", "seventy-five"]),
]

TASKS = ([("recall", q, a) for q, a in RECALL]
         + [("arith", q, a) for q, a in ARITH]
         + [("context", q, a) for q, a in CONTEXT])

if __name__ == "__main__":
    from collections import Counter
    print(f"total tasks: {len(TASKS)}")
    for k, v in Counter(s for s, _, _ in TASKS).items():
        print(f"  {k:<8} {v}")
