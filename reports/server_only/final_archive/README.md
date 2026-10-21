# SERVER EXIT 最終退避

`GITHUB_EXPORT_VERIFIED=PASS`、`PAPER_ASSETS_READY=PASS`、`USER_PC_COPY=AWAITING_USER_COPY`。

新しい未公開Draft Release **419216573** に、既存研究データ・入力・GT・raw・評価・解析・Git履歴を保存した。全14assetをGitHubのasset IDから同じサーバー上の新しい空directoryへ再取得し、全assetと全29,857memberのsize/SHAを照合した。元パスへのfallbackは使っていない。利用者のWindows PCへのコピー確認はまだ行っていない。

## 保存範囲と配置

- データsnapshot: `0647b012b9a163143713078e5c236efdb52825ec`。専用branchは `research/server-only-20261013`。mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` のまま。
- 元71,644パスを一度のinventory/hash passで固定し、新規の執筆索引・D転記4ファイルを追加。計71,648パス、重複除去後28,852data objects。必須参照23,096パスに未解決0件。
- `final-restore-001.tar.gz`〜`010.tar.gz` は独立したgzip tar。全partが必要。原本bytesは `objects/<sha256>`、旧絶対pathとの対応は `metadata/ORIGINAL_PATH_MAP.json`、各取得先は `metadata/COVERAGE.json`。元JSON・署名・hashを書き換えていない。
- 60建築・180description・360採点画像、先行160・継続781試行・補完3/17、初期200建築と旧1200出力、Goal007〜014の全保存解析と特徴/CI draw、親S/F72送信＋未送信8計画、C12/D16を含む。旧Releaseは来歴として保持し、復元に旧assetの追加取得は不要。
- Git履歴は完全bundleを4segmentで保持。復元後の新規bare repositoryでもrefs・全object・fsck・旧`.gitignore`の実測hashを検証した。現行版と異なる旧`.gitignore`の取得先は[履歴版coverage](HISTORICAL_GIT_COVERAGE.json)に分けた。
- `PREEXISTING_MISSING`は114**記録**であり114ファイルではない。100個の旧Gemini logical IDs、8 module paths、5集約artifact sets、40削除済みdirectoryを表す1集約記録を保持した。生成で補っていない。
- 外部ソフト依存は27項目（公式checkpoint/wheel23項目、runtime/環境4項目）。URL・版・配布hash・利用条件は `metadata/EXTERNAL_SOFTWARE_DEPENDENCIES.json`、実環境版は `metadata/ENVIRONMENT.json`。venv・OS・wheel・モデル重み・フォントは同梱しない。独自feature/scoreは同梱済み。
- public Gitへの追加はこの保存案内、Windows取得script、安全なD結果CSV/要約、索引、チェックサムと検証要約のみ。raw・入力・大容量dataはDraftにある。

全asset合計 **4,939,770,597 bytes（4.601 GiB）**。最大assetは851,618,930 bytesで、すべて1GiB以下。

## 執筆用素材

`paper-assets.tar.gz` は **32,914,150 bytes**、素材977ファイル（既存973＋新規4）を収録。検証後に単独で展開できる。

```powershell
if (Test-Path .\paper-assets) { throw 'Use a new paper-assets directory' }
mkdir .\paper-assets
tar -xzf .\minecraft-final-export-20261021\paper-assets.tar.gz -C .\paper-assets
if ($LASTEXITCODE -ne 0) { throw 'Paper extraction failed' }
```

索引は `paper-assets/paper/repository/reports/server_only/final_archive/WRITING_INDEX.md`。repository内の相対配置を保持し、canonical保存表は `paper/canonical/`、対応表は `PAPER_PATH_MAP.json` にある。主CSV/TeX/PNG/SVGとTeX参照画像はpaper packのmemberだけからも開けることを確認した。

[執筆索引](WRITING_INDEX.md)、[D実測16行](GOAL015D_RESULTS_16.csv)、[D日本語要約](GOAL015D_RESULTS_SUMMARY_JA.md)。DはA非空0/8・B8/8、Aの正常empty5件とtruncation3件を区別しNULLを保持した。Cは合成課題1種類、Dは実建築2件であり、本体60建築と合算しない。旧S/Fは未完・退化のまま、8未送信を完了にしていない。

## GitHub再取得と復元検査

[検証要約](VERIFICATION_SUMMARY.json)に全検査結果を保存した。主960、CLIP180予定、C12、D16、親S/F80予定行、代表raw・description・画像・GT・予測grid、主要paper素材を保存値のまま読んだ。53件の限定fixtureで危険なarchiveの拒否も確認した。全実験・採点・統計・旧全test suiteは再実行していない。

検証は同じサーバー上の `/root/minecraft-final-export-20261021T054443Z-9d421b47/github-restored` で実施した。既存Python3.12環境とNumPy/Pillowを利用し、画像とNPYをdecodeした。新しい環境へのinstall再現や別マシン検証を実施したという意味ではない。

手元に全assetを取得してSHAを確認した後、既存Python3.12以上とGitがある環境では次の読み取り専用復元が可能。output-rootはまだ存在しない場所を指定する。

```powershell
py -3 .\minecraft-final-export-20261021\restore.py --assets-dir .\minecraft-final-export-20261021 --output-root .\minecraft-restored
```

生成runner・launcher・保存済みauthorityは起動しない。元絶対pathはmapで解決し、旧サーバーの同名ファイルを参照しない。

## Windowsへの取得

[Windows手順](DOWNLOAD_WINDOWS.md)と[取得・Get-FileHash検証script](download-and-verify.ps1)を使用する。認証済みGitHub CLIは正確なasset IDから取得する。既知SSH alias `minecraft-research` の接続有効性は未確認。

短いSCP経路は、まだ存在しないdirectoryへ次を実行する。その後の短い検証コードはcatalogue自体の固定SHAと、全assetのサイズ/SHAを照合する。

```powershell
if (Test-Path .\minecraft-final-export-20261021) { throw 'Use a new destination directory' }
scp -r minecraft-research:/root/minecraft-final-export-20261021T054443Z-9d421b47/upload .\minecraft-final-export-20261021
if ($LASTEXITCODE -ne 0) { throw 'SCP failed' }
$d = '.\minecraft-final-export-20261021'
$c = Join-Path $d 'ARTIFACTS.json'
if ((Get-FileHash $c -Algorithm SHA256).Hash -ne '740083ce424c9d87d2b988ce915d6acbbc6597a5b92fd92ec34e46145232ac0e') { throw 'Catalogue SHA mismatch' }
$a = Get-Content $c -Raw | ConvertFrom-Json
foreach ($f in $a.assets) {
  $p = Join-Path $d $f.name
  if ((Get-Item $p).Length -ne $f.bytes -or (Get-FileHash $p -Algorithm SHA256).Hash -ne $f.sha256) { throw "Mismatch: $($f.name)" }
}
'ALL ASSETS VERIFIED ON THIS PC'
```

このPCでの検証成功を確認するまでは `USER_PC_COPY=AWAITING_USER_COPY` のまま。tokenをコマンドやファイルに埋め込まない。

## 正確な取得IDとchecksum

Release ID: **419216573**。tag: `server-exit-20261021T054443Z-9d421b47`。未公開Draftのため権限を持つGitHub認証が必要。tag URLやSource code.zipをデータの代用にしない。[機械可読catalogue](ARTIFACTS.json)は自己参照を避けて自身を一覧から除外しており、catalogue自身のID/SHAは以下に固定する。

| filename | asset ID | bytes | SHA256 |
|---|---:|---:|---|
| `EXPORT_MANIFEST.json` | 661546033 | 6,767,813 | `c01c1dc984dd931f4f59ce1e0ca192db9af39071d09d366208d7664e8e43f22e` |
| `final-restore-001.tar.gz` | 661546212 | 460,503,888 | `77276437246ae6314daacc6a3a13c3035db9f98cf69137a3b28b075327ede948` |
| `final-restore-002.tar.gz` | 661547276 | 456,527,218 | `359784d2da2cc6a7f9b3fe0655966cc29eee744d41f9ace9fb9edc445e3d8e1f` |
| `final-restore-003.tar.gz` | 661548712 | 328,199,161 | `4a913e6a849307399e65a69aaa36a8e43401ee8fa063a9cbb772c60781513fbb` |
| `final-restore-004.tar.gz` | 661549547 | 851,618,930 | `6f5ac21ccf9ab7fd49f68428e9180968672dc2e3673b2c9a31f6a6d0e9858e67` |
| `final-restore-005.tar.gz` | 661551550 | 358,112,367 | `741fd582bf23f14195b5ec469ea83e06aed40817fab95d97277003b8ccf22cd0` |
| `final-restore-006.tar.gz` | 661552395 | 477,001,668 | `0f7a32abf328e3ca6dfe177fd546cf6db2233d639e46df113930b7a270016cd5` |
| `final-restore-007.tar.gz` | 661554087 | 672,301,770 | `cdbf4eff82eb65e200006cd6a92dccb13a3d44505d7f47fcda0559096f3d4f40` |
| `final-restore-008.tar.gz` | 661556035 | 629,217,452 | `cc76f910d21a05bfe70ad1643b62312ab75fc04176bc1363d3649fcc2a6cbd77` |
| `final-restore-009.tar.gz` | 661558201 | 654,638,779 | `6d12ebb6fb32bd0069f7c7269c7e96703973d89df419776dccebf587e80f7d8f` |
| `final-restore-010.tar.gz` | 661560290 | 11,929,011 | `b1755fc86519d7e27c20e7a4a299dcc99581d247e2d93cca3c2e979f10b670ba` |
| `paper-assets.tar.gz` | 661560451 | 32,914,150 | `301e5dc03d6426b3a8063dbdf541656d175ef41fe04c70579b4ed921fe40a6fd` |
| `restore.py` | 661560634 | 35,498 | `149fc43f40934ea26d4f3bd1886605f28ff5db4e2664f46722fe7ccda0d8934b` |
| `ARTIFACTS.json` | 661560760 | 2,892 | `740083ce424c9d87d2b988ce915d6acbbc6597a5b92fd92ec34e46145232ac0e` |

## 状態と証跡の境界

データsnapshot commit、最終案内commit、この案内の外にあるpublication receiptを区別する。案内commitはこの文書を含むcommitであり、その実SHAとremote一致は最終回答および `/root/minecraft-final-export-20261021T054443Z-9d421b47/PUBLICATION_RECEIPT.json` に記録する。文書へ自身のcommit SHAを埋め込む循環参照は作らない。

元71,644パスのstat情報、旧tracked研究ファイル、旧結果/ledger、既存Draftの内容・assets、mainは不変。生成API送信0、追加推論0、人手実験0、旧未送信8件の実行0。サーバーの停止・再起動・削除・解約・課金操作は行っていない。
