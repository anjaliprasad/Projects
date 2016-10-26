lines = LOAD 'in' AS (line:chararray);
words = FOREACH lines GENERATE REPLACE(line,'', ' ') as word;
letters = FOREACH words GENERATE FLATTEN(TOKENIZE(LOWER(word))) as letter;
vowels = FOREACH (FILTER letters BY (letter == 'a') OR (letter == 'e') OR (letter == 'i') OR (letter == 'o') OR (letter == 'u')) GENERATE letter as vowel;
grouped = GROUP vowels BY vowel;
vowelcount = FOREACH grouped GENERATE group, COUNT(vowels);
final = FOREACH vowelcount GENERATE FLATTEN(($0,$1));
STORE final INTO 'vowelcount' USING PigStorage ('\t');
DUMP final;