from dataEngine.document_ir import IRBlock, parse, serialize

BLOCKS = [
    IRBlock("section-title", (153, 85, 770, 157), "विषय :- मारपीट\nछीन लेने के सम्बंध में।"),
    IRBlock("figure", (610, 879, 734, 918), ""),
    IRBlock("paragraph", (133, 178, 822, 940), "निवेदन पूर्वक कहना है"),
]


def test_round_trip():
    text = serialize(BLOCKS)
    assert text.startswith("<|ref|>section-title<|/ref|><|det|>[[153, 85, 770, 157]]<|/det|>\nविषय")
    assert parse(text) == (BLOCKS, True)


def test_base_model_style_output_with_several_boxes_uses_their_union():
    text = "<|ref|>text<|/ref|><|det|>[[10, 20, 30, 40], [5, 25, 50, 60]]<|/det|>\nab"
    blocks, valid = parse(text)
    assert valid and blocks == [IRBlock("text", (5, 20, 50, 60), "ab")]


def test_malformed_headers_are_skipped_and_flagged():
    text = (
        "<|ref|>text<|/ref|><|det|>[[10, 20, 5, 40]]<|/det|>\nbad box\n"  # x2 < x1
        "<|ref|>text<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>\ngood"
    )
    blocks, valid = parse(text)
    assert not valid
    assert blocks == [IRBlock("text", (1, 2, 3, 4), "good")]


def test_plain_text_is_not_structured_output():
    assert parse("just some text") == ([], False)
    assert parse("preamble <|ref|>t<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>\nx")[1] is False
