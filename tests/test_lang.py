import time

import pytest

from cbjev import lang


@pytest.mark.parametrize("text,code", [
    ("I was charged twice for my subscription, please refund me.", "en"),
    ("Can you help me reset my password? I tried twice.", "en"),
    ("Mein Konto wurde zweimal belastet, bitte erstatten Sie mir das Geld.", "de"),
    ("Bonjour, je veux annuler mon abonnement car il est trop cher.", "fr"),
    ("Hola, quiero cancelar mi suscripción porque es muy cara.", "es"),
    ("Olá, você pode me mandar a nota fiscal do meu pedido?", "pt"),
    ("Ciao, vorrei sapere dove è il mio ordine, grazie.", "it"),
    ("Ik wil mijn abonnement opzeggen want het is te duur.", "nl"),
    ("Dzień dobry, chcę anulować subskrypcję, bo jest za droga.", "pl"),
    ("Hej, jag vill säga upp mitt abonnemang för det är för dyrt.", "sv"),
    ("Hej, jeg vil gerne opsige mit abonnement, for det er for dyrt.", "da"),
    ("Hei, haluan peruuttaa tilaukseni koska se on liian kallis.", "fi"),
    ("Merhaba, aboneliğimi iptal etmek istiyorum çünkü çok pahalı.", "tr"),
    ("Bună ziua, vreau să anulez abonamentul pentru că este foarte scump.", "ro"),
    ("Dobrý den, chci zrušit předplatné, protože je příliš drahé.", "cs"),
    ("Szia, le szeretném mondani az előfizetésemet, mert nagyon drága.", "hu"),
    ("Halo, saya ingin membatalkan langganan saya karena terlalu mahal.", "id"),
    ("Xin chào, tôi muốn hủy đăng ký của tôi vì nó quá đắt.", "vi"),
])
def test_latin_languages(text, code):
    a = lang.analyse(text)
    assert a["script"] == "latin"
    assert a["language"] == code
    assert a["is_english"] is (code == "en")


@pytest.mark.parametrize("text,script", [
    ("Привет, я хочу отменить подписку.", "cyrillic"),
    ("Γεια σας, θέλω να ακυρώσω τη συνδρομή μου.", "greek"),
    ("أريد إلغاء اشتراكي من فضلك", "arabic"),
    ("אני רוצה לבטל את המנוי שלי", "hebrew"),
    ("मैं अपनी सदस्यता रद्द करना चाहता हूँ", "devanagari"),
    ("আমি আমার সদস্যতা বাতিল করতে চাই", "bengali"),
    ("ฉันต้องการยกเลิกการสมัครสมาชิก", "thai"),
    ("我想取消我的订阅", "han"),
    ("サブスクリプションをキャンセルしたい", "kana"),
    ("구독을 취소하고 싶어요", "hangul"),
])
def test_non_latin_scripts(text, script):
    a = lang.analyse(text)
    assert a["script"] == script
    assert a["is_english"] is False


def test_script_languages():
    assert lang.analyse("我想取消我的订阅")["language"] == "zh"
    assert lang.analyse("注文をキャンセルしたいです")["language"] == "ja"
    assert lang.analyse("구독을 취소하고 싶어요")["language"] == "ko"
    assert lang.analyse("Привет, я хочу отменить подписку.")["language"] is None   # ru/uk/bg...


@pytest.mark.parametrize("text", ["Hi", "ok thanks", "Quero cancelar", "12345", ""])
def test_short_text_is_undecided(text):
    assert lang.analyse(text)["language"] is None


def test_no_letters():
    a = lang.analyse("12345 !!! 0.5")
    assert a["script"] == "unknown" and a["language"] is None


def test_identifiers_ignored():
    # `com`, `o`, `e`, `da` would otherwise vote Portuguese/Italian
    a = lang.analyse("See o.e.com and da.com, write to me@o.com.br or https://www.de.die.das/und")
    assert a["language"] != "pt" and a["language"] != "it" and a["language"] != "de"


def test_english_loanwords_stay_english():
    for t in ("de facto en route to Rio de Janeiro", "Men at work: I was there at the end.",
              "I have a cat and a dog and a car in a box."):
        assert lang.analyse(t)["is_english"], t


def test_mixed_scripts():
    # a Latin plurality around a CJK request still reads as CJK
    assert lang.analyse("Order ABC-123456 for ACME Corp: 我想取消这个订单，请尽快退款")["script"] == "han"
    # a symbol and a capitalised name inside English prose do not
    a = lang.analyse("Please set α to 0.05 and email Дмитрий about it today.")
    assert a["script"] == "latin" and a["is_english"]


def test_state_structures():
    a = lang.analyse({"from": "a@b.c", "body": "Mein Konto wurde zweimal belastet, bitte helfen"})
    assert a["language"] == "de"
    assert lang.analyse([{"role": "user", "content": "我想退款"}])["script"] == "han"
    assert "from" not in lang.state_text({"from": "x"})


def test_accentless_unknown_language_goes_foreign():
    # no stopwords for the language, but plenty of foreign letters
    a = lang.analyse("Labas, noriu atšaukti prenumeratą, nes ji per brangi, ačiū už pagalbą")
    assert a["is_english"] is False


@pytest.mark.parametrize("code,want", [
    ("en", True), ("EN", True), ("en_US", True), ("en-GB", True), ("en_US.UTF-8", True), ("eng", True),
    ("English", True), ("de", False), ("pt_BR", False), ("zh-Hant", False), (None, None), ("", None), ("  ", None),
])
def test_english_code(code, want):
    assert lang.english_code(code) is want


def test_fast():
    text = "I was charged twice for my subscription, please refund me as soon as possible. " * 4
    lang.analyse(text)
    n = 500
    t = time.perf_counter()
    for _ in range(n):
        lang.analyse(text)
    assert (time.perf_counter() - t) / n < 1e-3
