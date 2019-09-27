def count_character(sentence):
    dictionary = {} 

    for character in sentence:
        keys = dictionary.keys()
        if character in keys:
                dictionary[character] += 1
        else:
            dictionary[character] = 1
    return dictionary

print(count_character('Hello'))