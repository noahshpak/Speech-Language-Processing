# @author: noahshpak
import enchant


"""
Naively parses input text into words following the max match pattern.
This algorithm is far better suited for Chinese since the language
consists of far shorter words on average than English.
"""
def parse(input: str, dictionary=enchant.Dict("en_US")):
    if not input: return []
    for i in reversed(range(len(input))):
        firstword = input[:i] # first i characters of input are our first check
        remainder = input[i:]
        if firstword and dictionary.check(firstword):
            return [firstword] + parse(remainder)
    else:
        firstword = input[0]
        remainder = input[1:]
        return [firstword] + parse(remainder)

