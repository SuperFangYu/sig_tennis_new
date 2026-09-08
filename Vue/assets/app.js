/** 网球动作分析页共享工具（正手/反手/发球/截击） */

function fileUrlForAction(relativePath, action) {
  if (!relativePath) return "";
  const q = encodeURIComponent(relativePath);
  const base = typeof window !== "undefined" && window.location ? window.location.origin : "";
  return `${base}/api/${action}/file?path=${q}`;
}

function parseCsvText(text) {
  const raw = String(text || "").replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (inQuotes) {
      if (ch === '"') {
        if (raw[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && raw[i + 1] === "\n") i++;
      row.push(cell);
      cell = "";
      if (row.length > 1 || (row.length === 1 && row[0] !== "")) rows.push(row);
      row = [];
    } else {
      cell += ch;
    }
  }
  row.push(cell);
  if (row.length > 1 || (row.length === 1 && row[0] !== "")) rows.push(row);
  if (!rows.length) return { headers: [], records: [] };
  const headers = rows[0].map((h) => String(h).trim());
  const records = rows.slice(1).map((cells) => {
    const rec = {};
    headers.forEach((h, idx) => {
      rec[h] = cells[idx] != null ? String(cells[idx]).trim() : "";
    });
    return rec;
  });
  return { headers, records };
}

function formatKinematicCell(val, decimals) {
  const d = decimals != null ? decimals : 2;
  if (val == null) return "-";
  const s = String(val).trim();
  if (!s || s.toLowerCase() === "nan" || s.toLowerCase() === "none" || s.toLowerCase() === "null") return "-";
  const n = Number(s);
  if (!Number.isNaN(n) && Number.isFinite(n)) return n.toFixed(d);
  return s || "-";
}

function segmentNumFromFilename(name) {
  const patterns = [
    /_angles_(\d+)\.png/i,
    /_rotation_(\d+)\.png/i,
    /_kinematics_(\d+)\.png/i,
    /_kinetic_chain_(\d+)\.png/i,
    /_speed_cog_(\d+)\.png/i,
  ];
  for (const re of patterns) {
    const m = String(name).match(re);
    if (m) return parseInt(m[1], 10);
  }
  return 0;
}

function artifactItemsByKind(artifacts, kind, fileUrlFn) {
  if (!Array.isArray(artifacts)) return [];
  const items = artifacts
    .filter((x) => x && x.kind === kind && x.relative_path)
    .map((x) => ({
      relative_path: x.relative_path,
      filename: x.filename || kind,
      src: fileUrlFn(x.relative_path),
    }));
  items.sort((a, b) => segmentNumFromFilename(a.filename) - segmentNumFromFilename(b.filename));
  return items;
}

const KINEMATIC_SUMMARY_COLUMNS = [
  { key: "segment_id", label: "片段编号", decimals: 0 },
  { key: "start_time", label: "起始时间(s)" },
  { key: "contact_time", label: "击球时间(s)" },
  { key: "end_time", label: "结束时间(s)" },
  { key: "duration", label: "动作时长(s)" },
  { key: "racket_head_speed_peak", label: "拍头速度峰值" },
  { key: "racket_elbow_angle_range", label: "肘角变化幅度" },
  { key: "racket_knee_angle_range", label: "膝角变化幅度" },
  { key: "racket_shoulder_angle_range", label: "肩角变化幅度" },
  { key: "racket_hip_angle_range", label: "髋角变化幅度" },
  { key: "shoulder_hip_angle_range", label: "肩髋角变化幅度" },
  { key: "racket_head_to_wrist_dist_peak", label: "拍头-手腕距离峰值" },
  { key: "racket_head_wrist_y_diff_at_contact", label: "拍头相对手腕高度" },
  { key: "quality_score", label: "质量评分" },
];

function createKinematicUiState() {
  return {
    kinematicSummaryOpen: true,
    kinematicSummaryRows: [],
    kinematicSummaryError: "",
    kinematicSummaryLoading: false,
    upperLimbBlockOpen: false,
    upperLimbRowOpen: {},
    lowerLimbBlockOpen: false,
    lowerLimbRowOpen: {},
    trunkBlockOpen: false,
    trunkRowOpen: {},
    racketKinBlockOpen: false,
    racketKinRowOpen: {},
  };
}

function createKinematicComputed(apiAction) {
  return {
    kinematicSummaryColumns() {
      if (!this.kinematicSummaryRows.length) return KINEMATIC_SUMMARY_COLUMNS;
      const keys = new Set(Object.keys(this.kinematicSummaryRows[0] || {}));
      return KINEMATIC_SUMMARY_COLUMNS.filter((col) => keys.has(col.key));
    },
    kinematicSummaryCsvItem() {
      const r = this.result;
      if (!r || !Array.isArray(r.artifacts)) return null;
      return r.artifacts.find((x) => x && x.kind === "kinematic_summary_csv" && x.relative_path) || null;
    },
    kinematicPhaseCsvItem() {
      const r = this.result;
      if (!r || !Array.isArray(r.artifacts)) return null;
      return r.artifacts.find((x) => x && x.kind === "kinematic_phase_summary_csv" && x.relative_path) || null;
    },
    upperLimbChartItems() {
      return artifactItemsByKind(this.result && this.result.artifacts, "upper_limb_angle_chart", (p) =>
        this.fileUrl(p)
      );
    },
    lowerLimbChartItems() {
      return artifactItemsByKind(this.result && this.result.artifacts, "lower_limb_angle_chart", (p) =>
        this.fileUrl(p)
      );
    },
    trunkRotationChartItems() {
      return artifactItemsByKind(this.result && this.result.artifacts, "trunk_rotation_chart", (p) =>
        this.fileUrl(p)
      );
    },
    racketKinematicChartItems() {
      return artifactItemsByKind(this.result && this.result.artifacts, "racket_kinematic_chart", (p) =>
        this.fileUrl(p)
      );
    },
  };
}

function createKinematicMethods() {
  return {
    formatCell(val, decimals) {
      return formatKinematicCell(val, decimals);
    },
    resetKinematicUi() {
      Object.assign(this, createKinematicUiState());
    },
    toggleChartRow(mapKey, i) {
      const map = this[mapKey] || {};
      this[mapKey] = { ...map, [i]: !map[i] };
    },
    async loadKinematicSummary() {
      this.kinematicSummaryRows = [];
      this.kinematicSummaryError = "";
      const item = this.kinematicSummaryCsvItem;
      if (!item) {
        this.kinematicSummaryError = "暂无运动学摘要";
        return;
      }
      this.kinematicSummaryLoading = true;
      try {
        const res = await fetch(this.fileUrl(item.relative_path));
        if (!res.ok) throw new Error("读取失败");
        const text = await res.text();
        const parsed = parseCsvText(text);
        this.kinematicSummaryRows = parsed.records || [];
        if (!this.kinematicSummaryRows.length) this.kinematicSummaryError = "暂无运动学摘要";
      } catch (e) {
        this.kinematicSummaryError = "暂无运动学摘要";
        console.warn("[kinematic summary]", e);
      } finally {
        this.kinematicSummaryLoading = false;
      }
    },
  };
}

function afterAnalyzeLoadKinematic(vm) {
  if (vm && typeof vm.loadKinematicSummary === "function") {
    vm.loadKinematicSummary();
  }
}

const KINEMATIC_CHART_PANELS = [
  {
    itemsKey: "upperLimbChartItems",
    blockKey: "upperLimbBlockOpen",
    rowKey: "upperLimbRowOpen",
    label: "上肢角度图（2D 图像平面）",
  },
  {
    itemsKey: "lowerLimbChartItems",
    blockKey: "lowerLimbBlockOpen",
    rowKey: "lowerLimbRowOpen",
    label: "下肢角度图（2D 图像平面）",
  },
  {
    itemsKey: "trunkRotationChartItems",
    blockKey: "trunkBlockOpen",
    rowKey: "trunkRowOpen",
    label: "躯干旋转图（2D 图像平面）",
  },
  {
    itemsKey: "racketKinematicChartItems",
    blockKey: "racketKinBlockOpen",
    rowKey: "racketKinRowOpen",
    label: "球拍运动学图",
  },
];

function createKinematicChartPanelsComputed() {
  return {
    kinematicChartPanels() {
      return KINEMATIC_CHART_PANELS.map((p) => ({
        ...p,
        items: this[p.itemsKey] || [],
        blockOpen: !!this[p.blockKey],
      })).filter((p) => p.items.length > 0);
    },
  };
}

function createKinematicChartPanelMethods() {
  return {
    toggleKinematicPanelBlock(blockKey) {
      this[blockKey] = !this[blockKey];
    },
    toggleKinematicPanelRow(rowKey, i) {
      const map = this[rowKey] || {};
      this[rowKey] = { ...map, [i]: !map[i] };
    },
    isPanelRowOpen(rowKey, i) {
      const map = this[rowKey];
      return !!(map && map[i]);
    },
  };
}

const ACTION_ANALYSIS_CONFIGS = {
  forehand: {
    action: "forehand",
    title: "正手击球",
    clipTitle: "切分后的正手片段",
    poseProgress: "身体姿态与特征…",
    exportProgress: "正手区间切分与导出…",
    panels: [
      { kind: "kinetic_chart", label: "动力链图", itemLabel: "片段" },
      { kind: "speed_cog_chart", label: "拍头速度与身体重心", itemLabel: "片段" },
    ],
  },
  backhand: {
    action: "backhand",
    title: "反手击球",
    clipTitle: "切分后的反手片段",
    poseProgress: "身体姿态与特征…",
    exportProgress: "反手区间切分与导出…",
    panels: [
      { kind: "kinetic_chart", label: "反手动力链图", itemLabel: "片段" },
      { kind: "speed_cog_chart", label: "速度与身体重心图", itemLabel: "片段" },
    ],
  },
  serve: {
    action: "serve",
    title: "发球",
    clipTitle: "切分后的发球片段",
    poseProgress: "身体姿态与发球特征…",
    exportProgress: "发球切分与图表导出…",
    panels: [
      { kind: "serve_trace_chart", label: "发球空间溯源与起跳", itemLabel: "图表" },
      { kind: "serve_kinetic_chart", label: "发球动力链与蓄力特征", itemLabel: "图表" },
    ],
  },
  volley: {
    action: "volley",
    title: "截击",
    clipTitle: "切分后的截击片段",
    poseProgress: "身体姿态与截击特征…",
    exportProgress: "截击切分与图表导出…",
    panels: [
      { kind: "volley_trace_chart", label: "截击速度突刺与平稳轨迹图", itemLabel: "图表" },
      { kind: "volley_kinetic_chart", label: "截击稳定结构动力链图", itemLabel: "图表" },
    ],
  },
};

function analysisConfigFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const action = String(params.get("action") || "forehand").toLowerCase();
  return ACTION_ANALYSIS_CONFIGS[action] || ACTION_ANALYSIS_CONFIGS.forehand;
}

function createActionAnalysisApp(config) {
  return {
    data() {
      return {
        config,
        selectedFile: null,
        busy: false,
        statusText: "等待上传",
        result: null,
        progressPercent: 0,
        progressTimer: null,
        progressHint: "就绪",
        handedness: "right",
        primaryPanelOpen: {},
        primaryPanelRowOpen: {},
        ...createKinematicUiState(),
      };
    },
    computed: {
      progressPctRounded() {
        return Math.round(this.progressPercent);
      },
      inputId() {
        return `analysis-file-${this.config.action}`;
      },
      clipItems() {
        return artifactItemsByKind(this.result && this.result.artifacts, "clip", (path) =>
          this.fileUrl(path)
        );
      },
      primaryPanels() {
        return this.config.panels
          .map((panel) => ({
            ...panel,
            items: artifactItemsByKind(
              this.result && this.result.artifacts,
              panel.kind,
              (path) => this.fileUrl(path)
            ),
            open: !!this.primaryPanelOpen[panel.kind],
          }))
          .filter((panel) => panel.items.length > 0);
      },
      ...createKinematicComputed(),
      ...createKinematicChartPanelsComputed(),
    },
    methods: {
      ...createKinematicMethods(),
      ...createKinematicChartPanelMethods(),
      fileUrl(relativePath) {
        return fileUrlForAction(relativePath, this.config.action);
      },
      clearProgressTimer() {
        if (this.progressTimer) {
          window.clearInterval(this.progressTimer);
          this.progressTimer = null;
        }
      },
      hintForProgress(percent) {
        if (percent < 18) return "正在上传…";
        if (percent < 45) return "球拍关键点追踪…";
        if (percent < 72) return this.config.poseProgress;
        if (percent < 92) return this.config.exportProgress;
        return "即将完成…";
      },
      startProgressSimulation() {
        this.clearProgressTimer();
        this.progressPercent = 3;
        this.progressHint = this.hintForProgress(this.progressPercent);
        this.progressTimer = window.setInterval(() => {
          if (this.progressPercent >= 96) return;
          const remaining = 96 - this.progressPercent;
          const step = Math.max(0.18, Math.min(1.7, remaining * 0.035));
          this.progressPercent = Math.min(96, this.progressPercent + step);
          this.progressHint = this.hintForProgress(this.progressPercent);
        }, 380);
      },
      resetPrimaryPanels() {
        this.primaryPanelOpen = {};
        this.primaryPanelRowOpen = {};
      },
      openPrimaryPanels() {
        const open = {};
        const rows = {};
        this.config.panels.forEach((panel) => {
          open[panel.kind] = true;
          rows[panel.kind] = { 0: true };
        });
        this.primaryPanelOpen = open;
        this.primaryPanelRowOpen = rows;
      },
      togglePrimaryPanel(kind) {
        this.primaryPanelOpen = {
          ...this.primaryPanelOpen,
          [kind]: !this.primaryPanelOpen[kind],
        };
      },
      togglePrimaryRow(kind, index) {
        const rows = this.primaryPanelRowOpen[kind] || {};
        this.primaryPanelRowOpen = {
          ...this.primaryPanelRowOpen,
          [kind]: { ...rows, [index]: !rows[index] },
        };
      },
      isPrimaryRowOpen(kind, index) {
        const rows = this.primaryPanelRowOpen[kind];
        return !!(rows && rows[index]);
      },
      resetForNewInput() {
        this.result = null;
        this.resetPrimaryPanels();
        this.resetKinematicUi();
        this.clearProgressTimer();
        this.progressPercent = 0;
        this.progressHint = "就绪";
      },
      onFile(event) {
        const file = event.target.files && event.target.files[0];
        this.selectedFile = file || null;
        this.resetForNewInput();
        this.statusText = file ? `已选择：${file.name}` : "等待上传";
      },
      async analyze() {
        if (!this.selectedFile || this.busy) return;
        this.busy = true;
        this.statusText = "上传并分析中，请稍候…";
        this.resetForNewInput();
        this.startProgressSimulation();

        const form = new FormData();
        form.append("file", this.selectedFile);
        form.append("handedness", this.handedness || "right");
        try {
          const response = await fetch(`/api/${this.config.action}/analyze`, {
            method: "POST",
            body: form,
          });
          if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            const detail = typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail || {});
            throw new Error(detail || response.statusText);
          }
          this.result = await response.json();
          const clipCount = this.clipItems.length;
          this.statusText =
            `完成 run_id: ${this.result.run_id != null ? this.result.run_id : "(未知)"}` +
            (clipCount ? `，共 ${clipCount} 段视频` : "");
          this.openPrimaryPanels();
          this.clearProgressTimer();
          this.progressPercent = 100;
          this.progressHint = "处理完成";
          afterAnalyzeLoadKinematic(this);
        } catch (error) {
          this.statusText = `失败：${error.message || error}`;
          this.clearProgressTimer();
          this.progressPercent = 0;
          this.progressHint = "处理失败";
        } finally {
          this.busy = false;
        }
      },
    },
    beforeUnmount() {
      this.clearProgressTimer();
    },
  };
}
