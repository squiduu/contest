from khaiii import KhaiiiApi

api = KhaiiiApi()

sentence = "안녕하세요, 저는 카카오 엔터테인먼트의 카이라고 합니다. 이건진짜좋은영화라라랜드진짜좋은영화"
analyzed = api.analyze(sentence)

for word in analyzed:
    print(word)
