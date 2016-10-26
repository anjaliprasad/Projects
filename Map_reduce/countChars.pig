lines = LOAD 'in' AS (line:chararray);
words = FOREACH lines GENERATE REPLACE(line,'', '|') as word;
letters = FOREACH words GENERATE FLATTEN(TOKENIZE(LOWER(word),'|')) as letter;
grouped = GROUP letters BY letter;
lettercount = FOREACH grouped GENERATE group, COUNT(letters);
letterswithoutspace = FILTER lettercount BY (group!=' ');
final = FOREACH letterswithoutspace GENERATE FLATTEN(($0,$1));
STORE final INTO 'charcount' USING PigStorage ('\t');
DUMP final;