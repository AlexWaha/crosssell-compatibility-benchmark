---
source_file: P-01_Journal_Article_NaukaiTekhnika_2026.pdf
relative_path: evidence/05-scholarly-articles/P-01_Journal_Article_NaukaiTekhnika_2026.pdf
source_sha256_short: afc822225e3b3448
doc_type: article
title: "Journal article in Nauka i Tekhnika Sohodni, No. 2(56) 2026: Method of Automated Verification of Technical Compatibility of Goods in Cross-Selling Systems Based on Hybrid Search and LLM"
date: 2026-02-25
date_kind: document_date
languages: [uk, en]
key_entities:
  organizations:
    - "Nauka i Tekhnika Sohodni (journal)"
    - "SolidProfessor"
    - "Ukrainian Assembly of Doctors of Sciences in Public Administration"
    - "Association of Scientists of Ukraine"
  people:
    - "Oleksandr Vitaliiovych Vakhovskyi (sole author)"
  locations:
    - "Mountain View, CA, USA"
    - "Ukraine"
key_facts:
  - "Sole author: Oleksandr Vitaliiovych Vakhovskyi, listed as Independent Researcher and Senior Software Engineer at SolidProfessor, Mountain View, CA, USA"
  - "ORCID 0009-0000-8183-0142 printed on article"
  - "DOI: 10.52058/2786-6025-2026-2(56)-1627-1642"
  - "Published in issue No. 2(56), year 2026"
  - "Page range 1627-1642 (16 pages)"
  - "ISSN 2786-6025 (Online) printed in page headers"
  - "Article submitted 12.02.2026; accepted after peer review 25.02.2026 (printed at end of article)"
  - "UDC 004.89:004.942:658.8"
  - "Article contains bilingual abstract (Ukrainian and English) plus full Ukrainian body and References section with 12 cited sources"
  - "Journal categorised by Ukrainian Ministry of Education as Category B across specialties 076 (Entrepreneurship and Trade), 015 (Professional Education), 081 (Law), 122 (Computer Sciences) - per acceptance certificate P-02"
criterion_relevance: [c05-scholarly-articles]
extraction:
  method: pdftotext
  confidence: high
  text_char_count: 44823
  pages: 16
  ocr_languages: []
  claude_vision_used: false
ocr_at: 2026-05-28
notes: ""
---

## Summary

Sole-authored peer-reviewed journal article published in issue 2(56) of "Nauka i Tekhnika Sohodni" in 2026. The article runs 16 pages (pp. 1627-1642) and carries DOI 10.52058/2786-6025-2026-2(56)-1627-1642 under ISSN 2786-6025. The author is listed as Oleksandr Vitaliiovych Vakhovskyi, Independent Researcher and Senior Software Engineer at SolidProfessor, Mountain View, CA, USA, ORCID 0009-0000-8183-0142. The article proposes a method for automated verification of technical compatibility of products in cross-selling systems, combining hybrid retrieval, retrieval-augmented generation with large language models, and formal logical verification. Submission date is 12 February 2026, acceptance after peer review is 25 February 2026.

## Key Facts

- Sole author printed on the article: "Ваховський Олександр Віталійович, незалежний дослідник, старший інженер з програмного забезпечення, SolidProfessor, Mountain View, CA, USA"
- ORCID 0009-0000-8183-0142 printed beside the author byline
- DOI 10.52058/2786-6025-2026-2(56)-1627-1642
- Issue 2(56), 2026; pages 1627-1642 (16 pages)
- ISSN 2786-6025 (Online) printed in every page header
- Submitted: 12.02.2026; accepted after peer review: 25.02.2026 (printed on final page)
- References list contains 12 cited works (IEEE Access, ACL 2025, Scientific Reports, Neural Computing and Applications, Electronics, NeurIPS, ICML, ACM Computing Surveys, Springer handbooks)
- Bilingual abstract (Ukrainian + English) printed at the start of the article
- Title (EN): "Method of Automated Verification of Technical Compatibility of Goods in Cross-Selling Systems Based on Hybrid Search and LLM"

## Raw Extracted Text

```
                                   № 2(56)
                              2026
ISSN 2786-6025 Online
       УДК 004.89:004.942:658.8
       https://doi.org/10.52058/2786-6025-2026-2(56)-1627-1642
       Ваховський Олександр Віталійович незалежний дослідник, старший
інженер з програмного забезпечення, SolidProfessor, Mountain View, CA, USA,
https://orcid.org/0009-0000-8183-0142

                  МЕТОД АВТОМАТИЗОВАНОЇ ВЕРИФІКАЦІЇ
             ТЕХНІЧНОЇ СУМІСНОСТІ ТОВАРІВ У СИСТЕМАХ
                 КРОС-ПРОДАЖІВ НА ОСНОВІ ГІБРИДНОГО
                                     ПОШУКУ ТА LLM

       Анотація. Актуальність дослідження зумовлена необхідністю підвищення достовірності рекомендаційних систем крос-продажів для технічно складних товарів, де помилки сумісності призводять до зростання операційних витрат і зниження довіри користувачів. Метою статті було обґрунтування методу автоматизованої верифікації технічної сумісності товарів на основі гібридного пошуку та великих мовних моделей із формальною логічною перевіркою технічних обмежень.

       Methodology, mathematical formalization, multi-level architecture (hybrid search + RAG + logical verification layer), credibility-control mechanisms (evidence-grounded generation, cross-source validation, probabilistic attribute reliability).

       Vakhovskyi Oleksandr Vitaliiovych Independent Researcher, Senior Software Engineer, SolidProfessor, Mountain View, CA, USA, https://orcid.org/0009-0000-8183-0142

                     METHOD OF AUTOMATED VERIFICATION
                   OF TECHNICAL COMPATIBILITY OF GOODS
                         IN CROSS-SELLING SYSTEMS BASED
                             ON HYBRID SEARCH AND LLM

       Abstract. The relevance of the study is due to the need to increase the reliability of cross-selling recommendation systems for technically complex products, where compatibility errors lead to increased operating costs and reduced user trust. The aim of the article was to substantiate the method of automated verification of technical compatibility of products based on hybrid search and large language models with formal logical verification of technical constraints.

       Keywords: recommender system, user trust, embedded vector spaces, large language models, logical verification.

       [Body sections cover: problem formulation; literature review citing Nawara & Kashef 2025 (IEEE Access), Wang et al. 2025 (ACL), Salau et al. 2025 (Scientific Reports), Abo El-Enen et al. 2025 (NCAA), Çiftlikçi et al. 2025 (Electronics), Althaf et al. 2025 (Computers); mathematical formalization with set I = {i1,…,iN}, attribute mapping xi: A -> V ∪ {⊥}, normalization operator η; binary compatibility variable yij; verification function V(i,j,c) = I[S(i,j) >= τS] · I[L(i,j,c) >= τL]; integral score(i,j,c) = α·S(i,j) + (1-α)·g(L(i,j,c)); evidence grounding q(t)=sim(v,z); cross-source validation via consensus mode; A/B testing experimental design with Precision@k, Recall@k, RMA rate, AOV, conversion rate metrics.]

       Література / References:
       1. Nawara D., Kashef R. (2025). IEEE Access, 13, 145772-145798. DOI: 10.1109/ACCESS.2025.3599832
       2. Wang S., Fan W., Feng Y., Lin S., Ma X., Wang S., Yin D. (2025). ACL 2025, 27152-27168. DOI: 10.18653/v1/2025.acl-long.1317
       3. Salau L., Mohamed H., Abdulsalam Y. (2025). Scientific Reports, 15, Article 13075. DOI: 10.1038/s41598-025-97407-3
       4. Abo El-Enen M., Saad S., Nazmy T. (2025). Neural Computing and Applications, 37, 28191-28267. DOI: 10.1007/s00521-025-11666-9
       5. Çiftlikçi M. S. et al. (2025). Electronics, 14(10), 1930. DOI: 10.3390/electronics14101930
       6. Althaf A. M. et al. (2025). Computers, 14(12), 525. DOI: 10.3390/computers14120525
       7. Aggarwal C. C. (2016). Recommender Systems: The Textbook. Springer.
       8. Zhang S., Yao L., Sun A., Tay Y. (2019). ACM Computing Surveys, 52(1), 1-38.
       9. Ricci F., Rokach L., Shapira B. (2022). Recommender Systems Handbook, 3rd ed. Springer.
       10. Lewis P. et al. (2020). NeurIPS 2020. arXiv:2005.11401
       11. Borgeaud S. et al. (2022). ICML 2022. arXiv:2112.04426
       12. Hogan A. et al. (2021). ACM Computing Surveys, 54(4), 1-37. DOI: 10.1145/3447772

       Дата першого надходження статті до видання: 12.02.2026
       Дата прийняття статті до друку після рецензування: 25.02.2026

[Full 44,823-char Ukrainian + English text available at .ocr-raw/P-01_Journal_Article_NaukaiTekhnika_2026.pdf.txt]
```
