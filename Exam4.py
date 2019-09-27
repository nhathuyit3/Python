def count_lower_upper(sentence):
    lower = 0
    upper = 0
    for character in sentence:
        if character.isupper():
            upper += 1
        elif character.islower():
            lower += 1
    return lower, upper

sentence = input('input you sentence')
lower, upper = count_lower_upper(sentence)
print(lower)
print(upper)