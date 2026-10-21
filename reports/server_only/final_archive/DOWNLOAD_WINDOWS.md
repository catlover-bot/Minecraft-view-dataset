# Windowsで取得・検証する

状態は **AWAITING_USER_COPY** です。サーバーとGitHubでの保存確認は、手元のPCへのコピー完了を意味しません。以下をWindows PCで実行し、全ファイルの検証が成功したことを確認してください。

固定した取得値：release ID `419216573`、`ARTIFACTS.json` asset ID `661560760`、catalogue SHA-256 `740083ce424c9d87d2b988ce915d6acbbc6597a5b92fd92ec34e46145232ac0e`。

このページと隣の[`download-and-verify.ps1`](download-and-verify.ps1)を、最終案内commitから保存します。この2ファイルはarchiveの外にある取得手順です。archiveをまだ展開する必要はありません。

## GitHub CLIで取得する

Windows PowerShell 5.1またはPowerShell 7で、既存のGitHub CLI認証を使います。非公開Draft releaseにアクセスできるアカウントが必要です。未認証なら通常の`gh auth login`を先に済ませてください。スクリプトへのtoken貼り付けは不要です。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\download-and-verify.ps1 -Source GitHub -Destination .\minecraft-final-export-20261021
```

スクリプトは固定asset IDでcatalogueを取得し、固定SHA-256を確認してから、catalogueに載る全assetを取得します。各ファイルのbyte数とSHA-256を照合し、名前・asset ID・byte数・省略しないSHA-256を表示します。既存の同一ファイルは再利用し、不一致ファイルは上書きせず停止します。

Windows PowerShellの`gh ... > file`はバイナリを壊すことがあるため使いません。このスクリプトはプロセスのバイナリstdoutをファイルに直接コピーします。取得に失敗した一時ファイルだけを削除し、既存の最終ファイルを保持します。失敗後は原因を確認し、同じコマンドで再開できます。

## SCPで取得する場合

SSH設定の`minecraft-research`が使えるPCでは、**まだ存在しない**保存先に次を実行します。既存ディレクトリへの`scp -r`は`upload`という階層を追加する場合があるため、既存のコピーと混ぜないでください。

```powershell
scp -r minecraft-research:/root/minecraft-final-export-20261021T054443Z-9d421b47/upload .\minecraft-final-export-20261021
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\download-and-verify.ps1 -Source Local -Destination .\minecraft-final-export-20261021
```

`Local`は通信せず、コピーされた`ARTIFACTS.json`の固定SHA-256と、そこに載る全assetのサイズ・SHA-256を照合します。catalogueが別の階層にある場合は、`-Destination`を**実際に`ARTIFACTS.json`があるディレクトリ**にしてください。

最後に`DOWNLOAD_AND_HASH_VERIFIED`が表示され、catalogueと全assetがPC上に残っていることを確認します。archive、manifest、検証記録をまとめて保持してください。復元は最終READMEの隔離復元手順に従います。このスクリプトは展開・実験・サーバー停止を実行しません。PC側の取得・検証が実際に済むまで、完了状態を変更しません。
