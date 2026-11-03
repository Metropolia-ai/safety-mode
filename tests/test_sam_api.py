"""Offline tests of the SAM client cache: no network, a fake OpenAI client."""
import io
from types import SimpleNamespace

from PIL import Image

from safety_mode.detect import SamClient


class FakeResponses:
    def __init__(self, text):
        self.text, self.calls = text, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(output_text=self.text)


def _jpeg(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


def test_cache_hit_makes_no_second_call(tmp_path):
    text = "<0f>0<|box;x1=10;y1=20;x2=29;y2=59;w=720;h=1280|><|mask;x=0;y=0;data=1,1,!!!!!!|>"
    fake = FakeResponses(text)
    c = SamClient(api_key="x", cache_dir=tmp_path, mask_encoding="one_bit", include_confidence=True)
    c._client = SimpleNamespace(responses=fake)
    img = _jpeg(1280, 2276)
    r1 = c.segment_image(img, "person")
    r2 = c.segment_image(img, "person")
    assert not r1.cached and r2.cached and c.calls == 1 and len(fake.calls) == 1
    kw = fake.calls[0]
    assert kw["model"] == "sam-3.1"
    assert kw["metadata"] == {"mask_encoding": "one_bit", "include_confidence": "true"}
    content = kw["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "person"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert r1.sent_size == (720, 1280) and r1.source_size == (1280, 2276)
    d = r1.detections(decode_masks=False)[0]
    assert d.box[0] == 10 * 1280 / 720 and d.box[3] == 60 * 2276 / 1280
    # another prompt is another request
    c.segment_image(img, "forklift")
    assert c.calls == 2
