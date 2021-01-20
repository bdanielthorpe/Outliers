from string import punctuation
import numpy as np
def numbers(datum):
    numbers = 0
    for char in datum:
        if char.isnumeric():
            numbers += 1
    return numbers

def letters(datum):
    letters = 0
    for char in datum:
        if char.isalpha():
            letters += 1
    return letters

punctuation = set(punctuation)
def specials(datum):
    specials = 0
    for char in datum:
        if char in punctuation:
            specials += 1
    return specials

vowels = {"a", "e", "i", "o", "u"}
def count_vowels(datum):
    num_vowels = 0
    for char in datum:
        if char in vowels:
            num_vowels += 1
    return num_vowels

def words(datum):
    return len(datum.split())

def distinct(datum):
    return len(set(datum))

datum_transformer = lambda datum: [
        numbers(datum), \
        letters(datum), \
        len(datum), \
        specials(datum), \
        words(datum), \
        count_vowels(datum), \
        distinct(datum) \
    ]

transform_data = lambda data: np.array([ datum_transformer(str(datum)) for datum in data])