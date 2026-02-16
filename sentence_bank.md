# Sentence Bank for Attention-Graph Analysis

This file provides reusable Norwegian (and a few English) test sentences for:
- pronoun and antecedent behavior,
- control/sub-verb constructions,
- subject/object and morphology coupling,
- lexical rarity and subtoken fragmentation effects.


## Usage

- Keep punctuation and casing stable between model runs.
- Run each sentence across all target models with identical settings.
- Log outputs using fields from `analysis_plan.md`.
- For minimal pairs, change one factor at a time.


## A) Pronoun and Antecedent (Coreference-like)

### A1. Basic antecedent candidates

1. `Eva sa at hun kom tidlig.`
2. `Ola sa at han kom tidlig.`
3. `Kari ringte Anne da hun kom hjem.`
4. `Per møtte Ola mens han ventet på bussen.`
5. `Maria takket Nora fordi hun hjalp henne.`
6. `Lars kritiserte Jon fordi han kom for sent.`
7. `Ola ga Per boka fordi han var sen.`
8. `Siri fortalte Eva at hun burde hvile.`
9. `Anne besøkte Kari etter at hun flyttet.`
10. `Nora skrev til Emma da hun var i Bergen.`

### A2. Multi-sentence discourse pronouns

11. `Ola forsøkte å gi Eva en ring. Hun ville ikke ha noe fra ham.`
12. `Per møtte Jonas i går. Han virket stresset.`
13. `Ida snakket med Sara. Hun foreslo en ny plan.`
14. `Lise fant brevet. Det lå under stolen.`
15. `Erik mistet nøkkelen. Den lå i jakken hele tiden.`

### A3. Ambiguity stress tests

16. `Kari sa til Anne at hun måtte dra tidlig.`
17. `Per fortalte Ole at han burde ringe moren sin.`
18. `Nora så Maria før hun gikk på toget.`
19. `Jon møtte Peter etter at han hadde spist.`
20. `Eva fortalte Ida at hun hadde bestått prøven.`


## B) Control and Sub-Verb Constructions

### B1. Subject control

21. `Han prøvde å sykle.`
22. `Hun lovet å komme i morgen.`
23. `Per forsøkte å løse oppgaven.`
24. `Kari planla å reise alene.`
25. `Ola håpet å vinne kampen.`
26. `Mina bestemte seg for å bli hjemme.`

### B2. Object control and contrasts

27. `Hun ba ham dra tidlig.`
28. `Per overtalte Ola til å bli med.`
29. `Kari tvang ham til å forklare seg.`
30. `Læreren ba elevene jobbe videre.`
31. `Siri oppfordret Eva til å søke stillingen.`

### B3. Near-control / raising-like variants

32. `Det ser ut til å regne i kveld.`
33. `Han virker å være sliten.`
34. `Hun ser ut til å forstå problemet.`
35. `Det begynte å snø i natt.`


## C) Subject, Object, and Morphology

### C1. Active vs passive

36. `Han spiste eplet.`
37. `Eplet ble spist av ham.`
38. `Hun skrev brevet.`
39. `Brevet ble skrevet av henne.`
40. `De reparerte bilen.`
41. `Bilen ble reparert av dem.`

### C2. Verb inflection visibility

42. `Han likte filmen.`
43. `Han liker filmen.`
44. `Han vil like filmen.`
45. `Hun krevde svar.`
46. `Hun krever svar.`
47. `De fant løsningen.`
48. `De finner løsningen.`

### C3. Subject/object minimal shifts

49. `Ola slo Per.`
50. `Per slo Ola.`
51. `Hun ga ham boka.`
52. `Han ga henne boka.`
53. `Læreren roste eleven.`
54. `Eleven roste læreren.`


## D) Lexical Rarity and Subtoken Fragmentation

### D1. Compound word contrasts

55. `Han fant en sykkel i båthuset.`
56. `Han fant en sykkel i parken.`
57. `Hun la nøkkelen i hanskerommet.`
58. `Hun la nøkkelen i skuffen.`
59. `De møttes ved trikkestoppet.`
60. `De møttes ved stasjonen.`

### D2. Rare vs frequent lexical alternatives

61. `Hun undersøkte skriftbildet nøye.`
62. `Hun undersøkte teksten nøye.`
63. `De diskuterte årsaksforholdet grundig.`
64. `De diskuterte årsaken grundig.`
65. `Han beskrev framgangsmåten tydelig.`
66. `Han beskrev metoden tydelig.`

### D3. Morphological tail behavior

67. `Han kjøpte bilen.`
68. `Han kjøpte en bil.`
69. `Hun så katten i hagen.`
70. `Hun så en katt i hagen.`


## E) Pronoun + Control Combined

71. `Eva ba Ola prøve å hjelpe henne.`
72. `Ola lovet Eva å hjelpe henne.`
73. `Kari ba Per forsøke å ringe henne.`
74. `Per lovet Kari å ringe henne.`
75. `Nora sa at hun prøvde å forstå ham.`


## F) English Calibration Set (optional)

Use these to compare tokenizer and structure behavior:

76. `John ate an apple at breakfast.`
77. `John tried to bike.`
78. `Mary told Anna that she was late.`
79. `Paul gave Peter a ring because he insisted.`
80. `He promised to help her.`


## Suggested Batches

- Quick smoke run: 21, 27, 36, 55, 76
- Pronoun-focused run: 1-20
- Control-focused run: 21-35, 71-75
- Morphology-focused run: 36-54, 67-70
- Fragmentation-focused run: 55-66
