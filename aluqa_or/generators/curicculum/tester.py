from aluqa_or.generators.curicculum.class_a import CuricculumClassA
from aluqa_or.generators.curicculum.class_b import CuricculumClassB
from aluqa_or.generators.curicculum.class_c import CuricculumClassC
from aluqa_or.generators.curicculum.class_d import CuricculumClassD
from aluqa_or.generators.curicculum.class_e import CuricculumClassE
from aluqa_or.generators.curicculum.class_f import CuricculumClassF


passage = "Coming off their impressive road win over the 49ers, " \
          "the Falcons went home for a Week 6 Sunday night duel with the Chicago Bears. " \
          "After a scoreless first quarter, Atlanta would trail early in the second quarter" \
          " as Bears quarterback Jay Cutler found wide receiver Johnny Knox on a 23-yard " \
          "touchdown pass. Afterwards, the Falcons took the lead as quarterback Matt Ryan " \
          "completed a 40-yard touchdown pass to wide receiver Roddy White and a 10-yard" \
          " touchdown pass to tight end Tony Gonzalez. After a scoreless third quarter, " \
          "Chicago would tie the game in the fourth quarter with Cutler hooking up with " \
          "tight end Greg Olsen on a 2-yard touchdown. Atlanta would regain the lead as " \
          "running back Michael Turner got a 5-yard touchdown run. Afterwards, the defense" \
          " would fend off a last-second Bears drive to lock up the victory."

# General
# letters_freq_in_passage = extract_letters_frequency(passage)
# letters_freq_in_first_sentence = letters_freq = extract_letters_frequency(passage, sentence_idx=0)
# all_numbers = extract_passage_numbers(passage)
#
# sentences = extract_sentences(passage)
# all_names_in_first_sentence = extract_ner(sentence=sentences[0])
# all_nouns_in_first_sentence = extract_pos(sentence=sentences[0], pos_to_retrieve=['NN'])


# Class A
# print(classA.how_many_times_character_appears(passage))
# print(classA.how_many_words_in_total(passage))
# print(classA.how_many_sentences_in_total(passage))
# print(classA.how_many_title_case_words_in_total(passage))
# print('Class A samples:')
# print('-----------------')
# classA = CuricculumClassA()
# for i in range(20):
#     print(classA.sample(passage))
# print()

# print('Class B samples:')
# print('-----------------')
# classB = CuricculumClassB()
# for i in range(20):
#     print(classB.sample(passage))
# print()

# print('Class C samples:')
# print('-----------------')
# classC = CuricculumClassC()
# for i in range(20):
#     print(classC.sample(passage))
# print()

# print('Class D samples:')
# print('-----------------')
# classD = CuricculumClassD()
# for i in range(20):
#     print(classD.sample(passage))
# print()

# print('Class E samples:')
# print('-----------------')
# classE = CuricculumClassE()
# for i in range(20):
#     print(classE.sample(passage))
# print()

print('Class F samples:')
print('-----------------')
classF = CuricculumClassF()
for i in range(20):
    print(classF.sample(passage))
print()
