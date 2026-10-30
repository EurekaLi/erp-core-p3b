"""命令行入口：复用冻结ICA重建长TFR epochs，先pilot再通过门控运行全体。"""

from erp_core_p3b.tfr_epochs import main

if __name__ == "__main__":
    raise SystemExit(main())
