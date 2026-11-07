from docx import Document
from pathlib import Path
import json
root=Path('/Users/omerfarukozturk/Desktop/turkce-quantization/outputs/wos_audit')
src=Path('/Users/omerfarukozturk/Desktop/Effects_of_Weight_Quantization_on_Turkish_Morphosyntactic_Performance_in_EuroLLM.docx')
doc=Document(src)
changes=[]
replacements={
'Running language models on local hardware': 'Running language models on local hardware requires sufficient memory for both the weights and the computations performed during inference. Low-precision methods can reduce inference memory requirements, but their numerical designs differ. LLM.int8() combines 8-bit multiplication with a higher-precision path for outlier features, whereas SmoothQuant transforms weights and activations to support W8A8 inference (Dettmers et al., 2022; Xiao et al., 2023). These efficiency results motivate a separate empirical question: whether quantization preserves a model’s preferences on specific grammatical contrasts.',
'In Turkish, grammatical relations': 'In Turkish, grammatical relations such as agreement, case selection, and subordination are often expressed through morphology. Tokenization is therefore relevant to how linguistic information is presented to a model. Toraman et al. (2023) compared five tokenization approaches, including a morphological-level tokenizer, in Turkish language models evaluated on six downstream tasks. Their study motivates attention to tokenization choices, but it does not establish how a fixed tokenizer interacts with weight quantization in minimal-pair judgments. A decline in accuracy alone does not identify the grammatical or representational cause.',
'Quantization schemes with the same nominal': 'Quantization schemes with the same nominal bit width can implement different numerical operations. LLM.int8() uses vector-wise quantization for most features and a 16-bit computation path for outlier dimensions (Dettmers et al., 2022). SmoothQuant uses a mathematically equivalent transformation to shift quantization difficulty from activations to weights, enabling 8-bit quantization of both (Xiao et al., 2023). These methods illustrate why bit width alone does not specify a quantization procedure. The present comparison holds the MLX affine scheme and group size fixed and varies the weight bit width. No further training or task-specific fine-tuning was performed.',
'2.2. Multilingual and Turkish evaluations':'2.2. Efficiency and Turkish tokenization',
'Marchisio et al. found': 'Published efficiency gains depend on the evaluated model, numerical method, and workload. Dettmers et al. (2022) report memory savings for mixed-precision 8-bit inference, while Xiao et al. (2023) evaluate memory and speed for W8A8 inference. Neither result establishes preservation of Turkish morphosyntactic preferences under MLX affine weight quantization. For Turkish, Toraman et al. (2023) show that tokenization granularity and vocabulary size warrant empirical evaluation across downstream tasks. Our study connects these concerns by measuring grammatical preferences and local resource use while holding the tokenizer fixed. Differences in tasks and quantization procedures preclude direct numerical rankings across these studies.',
'BLiMP uses minimally contrasting': 'BLiMP uses minimally contrasting sentences to probe language models’ sensitivity to grammatical distinctions in English (Warstadt et al., 2020). TurBLiMP provides the Turkish minimal pairs used in the present experiment (Başar et al., 2025). Turkish tokenization research compares character-, subword-, word-, and morphology-related representations, with evaluations on downstream tasks rather than the quantized minimal-pair task studied here (Toraman et al., 2023). We treat minimal-pair accuracy as sensitivity to the selected contrasts. Our token-to-word ratio is a sentence-level fragmentation proxy; it does not identify morphemes or measure alignment with gold morphological boundaries.',
'The fragmentation results complement': 'The fragmentation analysis addresses a different question from comparisons of alternative Turkish tokenizers (Toraman et al., 2023): here, the tokenizer is fixed and only weight precision changes. A word split into several tokens need not represent a difficult grammatical contrast. Punctuation contributes to the numerator of the proxy, and length enters both the ratio and the adjustment model. The token-length sensitivity interval includes the neutral value, which limits any interpretation of the negative slope as a robust linguistic effect.'
}
for p in list(doc.paragraphs):
    for start,new in replacements.items():
        if p.text.startswith(start):
            old=p.text
            # Keep paragraph properties; only replace affected prose runs.
            for child in list(p._p):
                if child.tag.endswith('}pPr'): continue
                p._p.remove(child)
            p.add_run(new)
            changes.append({'before':old,'after':new})
            break
remove=('Bayram, C.', 'Başar, E., & Bisazza', 'Frantar, E.', 'Lin, J., Tang', 'Marchisio, K.')
for p in list(doc.paragraphs):
    if p.text.startswith(remove):
        changes.append({'removed_reference':p.text}); p._element.getparent().remove(p._element)
for p in doc.paragraphs:
    if p.text.startswith('UTTER Project'):
        n=doc.add_paragraph('Toraman, C., Yilmaz, E. H., Şahinuç, F., & Ozcelik, O. (2023). Impact of Tokenization on Language Models: An Analysis for Turkish. ACM Transactions on Asian and Low-Resource Language Information Processing, 22(4), Article 116, 1–21. https://doi.org/10.1145/3578707',style=p.style)
        # alphabetical placement before UTTER
        p._p.addprevious(n._p)
        break
out=root/'EuroLLM_Turkish_WoS_revised.docx'
doc.save(out)
json.dump(changes,open(root/'changes.json','w'),ensure_ascii=False,indent=2)
# Ensure all experiment content, tables, math and images are retained.
orig=Document(src)
assert len(orig.tables)==len(doc.tables)
assert [t._tbl.xml for t in orig.tables]==[t._tbl.xml for t in doc.tables]
from lxml import etree
ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
for tag in ('m:oMath','w:drawing'):
    assert [etree.tostring(x) for x in orig._element.xpath('.//'+tag)]==[etree.tostring(x) for x in doc._element.xpath('.//'+tag)]
# Sections 3–7, 9–10, abstract and transparency were not edited.
for p in orig.paragraphs:
    if p.text and p.text not in [x.get('before') for x in changes] and not p.text.startswith(remove):
        assert p.text in [q.text for q in doc.paragraphs],p.text[:80]
print(out)
print('Tables, equations, drawings and all unaffected paragraphs preserved; changed prose paragraphs:',len(replacements))
