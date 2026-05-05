package require Tk

set app_title "动作分析工作台"
set project_root [file normalize [file join [file dirname [info script]] ".."]]
set runtime_root [file join $project_root ".desktop_runtime"]
file mkdir $runtime_root

set python_bin [lindex $argv 0]
set app_py [lindex $argv 1]
set default_weights [lindex $argv 2]
set default_template [lindex $argv 3]
set default_outputs [lindex $argv 4]

set selected_path ""
set current_task_id ""
set preview_after_id ""
set state_after_id ""
set task_pid ""
set browser_entries {}

wm title . $app_title
wm geometry . 1380x900
wm minsize . 1160 760
option add *Font {PingFang\ SC 12}
. configure -bg "#eef3f8"

array set form {
    source_mode video
    video_path ""
    description_file ""
    description_text "八段锦动作，节奏缓慢，双臂抬起保持对称，躯干保持直立，关键姿态有短暂停顿。"
    weights_path ""
    template_file ""
    frame_stride 2
    max_frames 120
    camera_index 0
    camera_max_frames 180
    browser_path ""
    status "等待开始分析"
    summary "这里会显示分析摘要、状态或错误信息。"
    output_dir ""
    progress 0
}
set form(weights_path) $default_weights
set form(template_file) $default_template
set form(browser_path) $project_root
set form(output_dir) $default_outputs

array set camera_map {}
set camera_labels {}

proc make_header {parent title subtitle} {
    frame $parent -bg "#0b3c5d" -height 84
    pack $parent -fill x
    pack propagate $parent 0
    label $parent.title -text $title -bg "#0b3c5d" -fg "white" -font {PingFang SC 24 bold}
    label $parent.sub -text $subtitle -bg "#0b3c5d" -fg "#d6e8fa" -font {PingFang SC 11}
    place $parent.title -x 24 -y 16
    place $parent.sub -x 24 -y 52
}

proc style_panel {widget} {
    $widget configure -bg "white" -highlightthickness 1 -highlightbackground "#d4dce6"
}

proc add_field_title {parent text} {
    label $parent -text $text -bg "white" -fg "#24384d" -font {PingFang SC 11 bold}
}

proc set_status {text} {
    global form
    set form(status) $text
}

proc set_summary {text} {
    .right.summary.text delete 1.0 end
    .right.summary.text insert 1.0 $text
}

proc set_progress {value} {
    global form
    set form(progress) $value
    .bottom.progress configure -value $value
}

proc update_selected_path_label {} {
    global selected_path
    if {$selected_path eq ""} {
        .left.browser.selected configure -text "当前未选中文件"
    } else {
        .left.browser.selected configure -text "当前选择：$selected_path"
    }
}

proc browse_directory {{path ""}} {
    global form selected_path
    if {$path eq ""} {
        set path $form(browser_path)
    }
    if {$path eq ""} {
        set path [pwd]
    }
    set path [file normalize $path]
    if {![file exists $path]} {
        tk_messageBox -icon error -title "目录不存在" -message "目录不存在：$path"
        return
    }
    if {[file isfile $path]} {
        set path [file dirname $path]
    }
    set form(browser_path) $path
    .left.browser.path delete 0 end
    .left.browser.path insert 0 $path
    .left.browser.list delete 0 end

    set entries [glob -nocomplain -directory $path * .*]
    set cleaned {}
    foreach item $entries {
        set name [file tail $item]
        if {$name in {. ..}} {
            continue
        }
        lappend cleaned $item
    }
    set dirs {}
    set files {}
    foreach item [lsort -dictionary $cleaned] {
        if {[file isdirectory $item]} {
            lappend dirs $item
        } else {
            lappend files $item
        }
    }
    foreach item [concat $dirs $files] {
        set name [file tail $item]
        if {[file isdirectory $item]} {
            .left.browser.list insert end "📁 $name"
        } else {
            .left.browser.list insert end "📄 $name"
        }
    }
    set ::browser_entries [concat $dirs $files]
    set selected_path ""
    update_selected_path_label
}

proc browser_select {} {
    global browser_entries selected_path
    set sel [.left.browser.list curselection]
    if {$sel eq ""} {
        return
    }
    set index [lindex $sel 0]
    set selected_path [lindex $browser_entries $index]
    update_selected_path_label
    if {[file isfile $selected_path]} {
        set lower [string tolower $selected_path]
        if {[regexp {(\.mp4|\.mov|\.avi|\.mkv)$} $lower]} {
            set ::form(video_path) $selected_path
            update_video_preview
        } elseif {[regexp {(\.txt|\.md|\.json)$} $lower]} {
            set ::form(description_file) $selected_path
        } elseif {[regexp {(\.pt)$} $lower]} {
            set ::form(weights_path) $selected_path
        }
    }
}

proc browser_open {} {
    global browser_entries
    set sel [.left.browser.list curselection]
    if {$sel eq ""} {
        return
    }
    set index [lindex $sel 0]
    set target [lindex $browser_entries $index]
    if {[file isdirectory $target]} {
        browse_directory $target
        return
    }
    browser_select
    preview_selected_file
}

proc browser_up {} {
    global form
    set current $form(browser_path)
    if {$current eq ""} {
        return
    }
    browse_directory [file dirname $current]
}

proc pick_video {} {
    set path [tk_getOpenFile -title "选择视频文件" -filetypes {
        {"视频文件" {.mp4 .mov .avi .mkv}}
        {"全部文件" *}
    }]
    if {$path ne ""} {
        set ::form(video_path) [file normalize $path]
        update_video_preview
    }
}

proc pick_description {} {
    set path [tk_getOpenFile -title "选择描述文本文件" -filetypes {
        {"文本文件" {.txt .md .json}}
        {"全部文件" *}
    }]
    if {$path ne ""} {
        set ::form(description_file) [file normalize $path]
        load_description_file
    }
}

proc pick_weights {} {
    set path [tk_getOpenFile -title "选择 YOLO 权重文件" -filetypes {
        {"PyTorch 权重" {.pt}}
        {"全部文件" *}
    }]
    if {$path ne ""} {
        set ::form(weights_path) [file normalize $path]
    }
}

proc pick_template {} {
    set path [tk_getOpenFile -title "选择动作模板文件" -filetypes {
        {"JSON 文件" {.json}}
        {"全部文件" *}
    }]
    if {$path ne ""} {
        set ::form(template_file) [file normalize $path]
    }
}

proc load_description_file {} {
    global python_bin app_py form
    if {$form(description_file) eq ""} {
        tk_messageBox -icon error -title "缺少描述文件" -message "请先选择描述文件。"
        return
    }
    if {![file exists $form(description_file)]} {
        tk_messageBox -icon error -title "文件不存在" -message "文件不存在：$form(description_file)"
        return
    }
    set cmd [list $python_bin $app_py --desktop-read-text $form(description_file)]
    if {[catch {set result [exec {*}$cmd]} result]} {
        tk_messageBox -icon error -title "读取失败" -message $result
        return
    }
    .left.description.text delete 1.0 end
    .left.description.text insert 1.0 $result
}

proc update_source_mode {} {
    global form
    if {$form(source_mode) eq "camera"} {
        grid .left.controls.cameraLabel -row 1 -column 0 -sticky w -padx 12 -pady 10
        grid .left.controls.cameraFrame -row 1 -column 1 -columnspan 2 -sticky ew -padx 8 -pady 10
        grid remove .left.controls.videoLabel
        grid remove .left.controls.videoFrame
    } else {
        grid .left.controls.videoLabel -row 1 -column 0 -sticky w -padx 12 -pady 10
        grid .left.controls.videoFrame -row 1 -column 1 -columnspan 2 -sticky ew -padx 8 -pady 10
        grid remove .left.controls.cameraLabel
        grid remove .left.controls.cameraFrame
    }
}

proc refresh_cameras {} {
    global python_bin app_py camera_map camera_labels form
    set cmd [list $python_bin $app_py --desktop-camera-scan]
    if {[catch {set result [exec {*}$cmd 2>@1]} result]} {
        tk_messageBox -icon error -title "摄像头扫描失败" -message $result
        return
    }
    array unset camera_map
    set camera_labels {}
    foreach line [split $result "\n"] {
        if {[string trim $line] eq ""} {
            continue
        }
        if {![regexp {^([0-9]+)\t([01])$} $line -> idx readable]} {
            continue
        }
        set label "摄像头 $idx"
        if {$readable eq "1"} {
            append label "（可读取）"
        } else {
            append label "（已发现）"
        }
        set camera_map($label) $idx
        lappend camera_labels $label
    }
    if {[llength $camera_labels] == 0} {
        set camera_labels {"摄像头 0（未检测到，默认使用）"}
        set camera_map("摄像头 0（未检测到，默认使用）") 0
    }
    set menu [.left.controls.cameraFrame.selector cget -menu]
    $menu delete 0 end
    foreach label $camera_labels {
        $menu add radiobutton -label $label -value $label -variable ::form(camera_label)
    }
    set form(camera_label) [lindex $camera_labels 0]
    apply_camera_selection
}

proc apply_camera_selection {} {
    global form camera_map
    if {[info exists camera_map($form(camera_label))]} {
        set form(camera_index) $camera_map($form(camera_label))
    }
}

proc build_request_file {task_id} {
    global form runtime_root
    set task_dir [file join $runtime_root $task_id]
    file mkdir $task_dir
    set request_path [file join $task_dir request.tsv]
    set handle [open $request_path w]
    fconfigure $handle -encoding utf-8 -translation lf
    dict set payload source_mode $form(source_mode)
    dict set payload video_path $form(video_path)
    dict set payload description_text [string trim [.left.description.text get 1.0 end]]
    dict set payload weights_path $form(weights_path)
    dict set payload template_file $form(template_file)
    dict set payload frame_stride $form(frame_stride)
    dict set payload max_frames $form(max_frames)
    dict set payload camera_index $form(camera_index)
    dict set payload camera_max_frames $form(camera_max_frames)
    dict set payload output_dir ""
    foreach {key value} $payload {
        set encoded [binary encode base64 [encoding convertto utf-8 $value]]
        puts $handle "$key\t$encoded"
    }
    close $handle
    return $task_dir
}

proc start_analysis {} {
    global form current_task_id task_pid runtime_root python_bin app_py state_after_id preview_after_id
    if {$current_task_id ne ""} {
        tk_messageBox -icon warning -title "任务进行中" -message "当前已有分析任务在运行，请先停止或等待完成。"
        return
    }
    set description [string trim [.left.description.text get 1.0 end]]
    if {$description eq ""} {
        tk_messageBox -icon error -title "缺少描述" -message "请先填写动作描述。"
        return
    }
    if {$form(source_mode) eq "video" && $form(video_path) eq ""} {
        tk_messageBox -icon error -title "缺少视频" -message "请先选择视频文件。"
        return
    }

    set task_id [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
    append task_id "_" [pid] "_" [clock clicks]
    set task_dir [build_request_file $task_id]
    set current_task_id $task_id
    set_status "正在分析..."
    set_summary "正在初始化分析器..."
    set_progress 0
    .bottom.start configure -state disabled
    .bottom.stop configure -state normal

    set cmd [list $python_bin $app_py --desktop-task [file join $task_dir request.tsv]]
    if {[catch {set task_pid [exec {*}$cmd &]} result]} {
        set current_task_id ""
        .bottom.start configure -state normal
        .bottom.stop configure -state disabled
        tk_messageBox -icon error -title "启动失败" -message $result
        return
    }
    poll_state
    poll_preview
}

proc stop_analysis {} {
    global current_task_id runtime_root
    if {$current_task_id eq ""} {
        return
    }
    set stop_path [file join $runtime_root $current_task_id stop.flag]
    set handle [open $stop_path w]
    puts $handle "stop"
    close $handle
    set_status "已请求停止"
}

proc read_state_file {path} {
    set data [dict create]
    if {![file exists $path]} {
        return $data
    }
    set handle [open $path r]
    fconfigure $handle -encoding utf-8
    while {[gets $handle line] >= 0} {
        if {$line eq ""} {
            continue
        }
        set pos [string first "\t" $line]
        if {$pos < 0} {
            continue
        }
        set key [string range $line 0 [expr {$pos - 1}]]
        set encoded [string range $line [expr {$pos + 1}] end]
        if {[catch {set value [encoding convertfrom utf-8 [binary decode base64 $encoded]]}]} {
            set value ""
        }
        dict set data $key $value
    }
    close $handle
    return $data
}

proc poll_state {} {
    global current_task_id runtime_root state_after_id
    if {$current_task_id eq ""} {
        return
    }
    set state_path [file join $runtime_root $current_task_id state.tsv]
    set data [read_state_file $state_path]
    if {[dict size $data] > 0} {
        if {[dict exists $data message]} {
            set_status [dict get $data message]
        }
        if {[dict exists $data summary_text]} {
            set_summary [dict get $data summary_text]
        }
        if {[dict exists $data progress]} {
            catch {set_progress [dict get $data progress]}
        }
        if {[dict exists $data output_dir]} {
            set ::form(output_dir) [dict get $data output_dir]
        }
        update_artifact_links $data
        if {[dict exists $data status]} {
            set status [dict get $data status]
            if {$status in {done error stopped}} {
                finish_task $status
                return
            }
        }
    }
    set state_after_id [after 600 poll_state]
}

proc finish_task {status} {
    global current_task_id preview_after_id state_after_id
    if {$state_after_id ne ""} {
        after cancel $state_after_id
        set state_after_id ""
    }
    if {$preview_after_id ne ""} {
        after cancel $preview_after_id
        set preview_after_id ""
    }
    set current_task_id ""
    .bottom.start configure -state normal
    .bottom.stop configure -state disabled
    if {$status eq "done"} {
        set_progress 100
        tk_messageBox -icon info -title "分析完成" -message "分析完成，结果已输出。"
    } elseif {$status eq "error"} {
        tk_messageBox -icon error -title "分析失败" -message "分析失败，请查看右侧摘要。"
    } else {
        tk_messageBox -icon info -title "已停止" -message "分析已停止。"
    }
}

proc poll_preview {} {
    global current_task_id runtime_root preview_after_id
    if {$current_task_id eq ""} {
        return
    }
    set preview_path [file join $runtime_root $current_task_id preview.png]
    if {[file exists $preview_path]} {
        update_image_widget .right.preview.canvas $preview_path
    }
    set preview_after_id [after 800 poll_preview]
}

proc update_image_widget {widget path} {
    if {![file exists $path]} {
        return
    }
    catch {image delete preview_img}
    image create photo preview_img -file $path
    $widget delete all
    $widget create image 10 10 -anchor nw -image preview_img
    set bbox [$widget bbox all]
    if {$bbox ne ""} {
        lassign $bbox x1 y1 x2 y2
        $widget configure -scrollregion [list 0 0 [expr {$x2 + 10}] [expr {$y2 + 10}]]
    }
}

proc update_video_preview {} {
    global form python_bin app_py runtime_root
    if {$form(video_path) eq "" || ![file exists $form(video_path)]} {
        return
    }
    set thumb_path [file join $runtime_root "video_preview.png"]
    set cmd [list $python_bin $app_py --desktop-video-thumb $form(video_path) $thumb_path]
    if {[catch {exec {*}$cmd}]} {
        return
    }
    update_image_widget .right.preview.canvas $thumb_path
}

proc preview_selected_file {} {
    global selected_path
    if {$selected_path eq ""} {
        return
    }
    if {[file isdirectory $selected_path]} {
        browse_directory $selected_path
        return
    }
    set lower [string tolower $selected_path]
    if {[regexp {(\.mp4|\.mov|\.avi|\.mkv)$} $lower]} {
        set ::form(video_path) $selected_path
        update_video_preview
    } elseif {[regexp {(\.txt|\.md|\.json)$} $lower]} {
        set ::form(description_file) $selected_path
        load_description_file
    }
}

proc open_output_dir {} {
    global form
    if {$form(output_dir) eq ""} {
        set dir [tk_chooseDirectory -title "选择输出目录"]
        if {$dir ne ""} {
            set form(output_dir) $dir
        }
        return
    }
    if {[file exists $form(output_dir)]} {
        exec open $form(output_dir) &
    }
}

proc open_artifact_path {path} {
    if {$path eq ""} {
        return
    }
    if {[file exists $path]} {
        exec open $path &
    }
}

proc update_artifact_links {data} {
    set mapping {
        analysis_summary_json .right.links.json
        frame_metrics_csv .right.links.csv
        overlay_video .right.links.video
        metrics_plot .right.links.plot
        summary_card .right.links.card
        preview_contact_sheet .right.links.sheet
    }
    foreach {key widget} $mapping {
        set dict_key "artifact_$key"
        if {[dict exists $data $dict_key]} {
            set path [dict get $data $dict_key]
            $widget configure -text $path -command [list open_artifact_path $path] -state normal
        } else {
            $widget configure -text "未生成" -command {} -state disabled
        }
    }
}

proc on_close {} {
    stop_analysis
    after 150
    destroy .
}

make_header .header $app_title "单独桌面弹窗版，不依赖当前 Python 的 tkinter。支持目录浏览、文件读取、摄像头、分析预览与结果打开。"

frame .content -bg "#eef3f8"
pack .content -fill both -expand 1 -padx 14 -pady 14

frame .left -bg "#eef3f8"
frame .right -bg "#eef3f8" -width 430
pack .left -in .content -side left -fill both -expand 1
pack .right -in .content -side right -fill y
pack propagate .right 0

frame .left.controls
style_panel .left.controls
pack .left.controls -fill x -pady {0 12}
grid columnconfigure .left.controls 1 -weight 1

add_field_title .left.controls.modeLabel "输入模式"
grid .left.controls.modeLabel -row 0 -column 0 -sticky w -padx 12 -pady 10
ttk::combobox .left.controls.mode -textvariable form(source_mode) -values {video camera} -state readonly
bind .left.controls.mode <<ComboboxSelected>> {update_source_mode}
grid .left.controls.mode -row 0 -column 1 -sticky ew -padx 8 -pady 10
button .left.controls.scanCamera -text "扫描摄像头" -command refresh_cameras -bg "#e8f1fb" -fg "#0b5cab" -relief flat
grid .left.controls.scanCamera -row 0 -column 2 -sticky ew -padx 8 -pady 10

add_field_title .left.controls.videoLabel "视频文件"
frame .left.controls.videoFrame -bg white
entry .left.controls.videoFrame.path -textvariable form(video_path) -relief flat -bg "#f7f9fb"
button .left.controls.videoFrame.pick -text "选择视频" -command pick_video -bg "#e8f1fb" -fg "#0b5cab" -relief flat
button .left.controls.videoFrame.preview -text "预览文件" -command update_video_preview -bg "#eef2f6" -fg "#33485f" -relief flat
pack .left.controls.videoFrame.path -side left -fill x -expand 1 -padx {0 8}
pack .left.controls.videoFrame.pick -side left -padx {0 8}
pack .left.controls.videoFrame.preview -side left

add_field_title .left.controls.cameraLabel "摄像头"
frame .left.controls.cameraFrame -bg white
set form(camera_label) "摄像头 0（默认）"
tk_optionMenu .left.controls.cameraFrame.selector form(camera_label) "摄像头 0（默认）"
button .left.controls.cameraFrame.apply -text "采用编号" -command apply_camera_selection -bg "#e8f1fb" -fg "#0b5cab" -relief flat
label .left.controls.cameraFrame.tip -textvariable form(camera_index) -bg "white" -fg "#5f6f82"
pack .left.controls.cameraFrame.selector -side left -fill x -expand 1 -padx {0 8}
pack .left.controls.cameraFrame.apply -side left -padx {0 8}
pack .left.controls.cameraFrame.tip -side left

grid .left.controls.videoLabel -row 1 -column 0 -sticky w -padx 12 -pady 10
grid .left.controls.videoFrame -row 1 -column 1 -columnspan 2 -sticky ew -padx 8 -pady 10

add_field_title .left.controls.weightsLabel "权重文件"
grid .left.controls.weightsLabel -row 2 -column 0 -sticky w -padx 12 -pady 10
frame .left.controls.weightsFrame -bg white
entry .left.controls.weightsFrame.path -textvariable form(weights_path) -relief flat -bg "#f7f9fb"
button .left.controls.weightsFrame.pick -text "选择权重" -command pick_weights -bg "#e8f1fb" -fg "#0b5cab" -relief flat
pack .left.controls.weightsFrame.path -side left -fill x -expand 1 -padx {0 8}
pack .left.controls.weightsFrame.pick -side left
grid .left.controls.weightsFrame -row 2 -column 1 -columnspan 2 -sticky ew -padx 8 -pady 10

add_field_title .left.controls.templateLabel "模板文件"
grid .left.controls.templateLabel -row 3 -column 0 -sticky w -padx 12 -pady 10
frame .left.controls.templateFrame -bg white
entry .left.controls.templateFrame.path -textvariable form(template_file) -relief flat -bg "#f7f9fb"
button .left.controls.templateFrame.pick -text "选择模板" -command pick_template -bg "#e8f1fb" -fg "#0b5cab" -relief flat
pack .left.controls.templateFrame.path -side left -fill x -expand 1 -padx {0 8}
pack .left.controls.templateFrame.pick -side left
grid .left.controls.templateFrame -row 3 -column 1 -columnspan 2 -sticky ew -padx 8 -pady 10

add_field_title .left.controls.paramsLabel "分析参数"
grid .left.controls.paramsLabel -row 4 -column 0 -sticky w -padx 12 -pady 10
frame .left.controls.paramsFrame -bg white
label .left.controls.paramsFrame.strideLabel -text "步长" -bg white
entry .left.controls.paramsFrame.stride -width 6 -textvariable form(frame_stride) -relief flat -bg "#f7f9fb"
label .left.controls.paramsFrame.maxLabel -text "最大帧数" -bg white
entry .left.controls.paramsFrame.max -width 8 -textvariable form(max_frames) -relief flat -bg "#f7f9fb"
label .left.controls.paramsFrame.camMaxLabel -text "摄像头帧数" -bg white
entry .left.controls.paramsFrame.camMax -width 8 -textvariable form(camera_max_frames) -relief flat -bg "#f7f9fb"
foreach widget {
    .left.controls.paramsFrame.strideLabel
    .left.controls.paramsFrame.stride
    .left.controls.paramsFrame.maxLabel
    .left.controls.paramsFrame.max
    .left.controls.paramsFrame.camMaxLabel
    .left.controls.paramsFrame.camMax
} {
    pack $widget -side left -padx 6 -pady 8
}
grid .left.controls.paramsFrame -row 4 -column 1 -columnspan 2 -sticky w -padx 8 -pady 10

frame .left.description
style_panel .left.description
pack .left.description -fill x -pady {0 12}
add_field_title .left.description.title "动作描述"
pack .left.description.title -anchor w -padx 12 -pady {12 6}
frame .left.description.fileRow -bg white
entry .left.description.fileRow.path -textvariable form(description_file) -relief flat -bg "#f7f9fb"
button .left.description.fileRow.pick -text "选择描述文件" -command pick_description -bg "#e8f1fb" -fg "#0b5cab" -relief flat
button .left.description.fileRow.load -text "读取文本" -command load_description_file -bg "#eef2f6" -fg "#33485f" -relief flat
pack .left.description.fileRow.path -side left -fill x -expand 1 -padx {12 8} -pady {0 8}
pack .left.description.fileRow.pick -side left -padx {0 8} -pady {0 8}
pack .left.description.fileRow.load -side left -padx {0 12} -pady {0 8}
pack .left.description.fileRow -fill x
text .left.description.text -height 7 -relief flat -bg "#f7f9fb" -wrap word
.left.description.text insert 1.0 $form(description_text)
pack .left.description.text -fill x -padx 12 -pady {0 12}

frame .left.browser
style_panel .left.browser
pack .left.browser -fill both -expand 1
add_field_title .left.browser.title "目录浏览器"
pack .left.browser.title -anchor w -padx 12 -pady {12 6}
frame .left.browser.pathRow -bg white
entry .left.browser.path -relief flat -bg "#f7f9fb"
button .left.browser.pathRow.load -text "读取目录" -command {browse_directory [.left.browser.path get]} -bg "#e8f1fb" -fg "#0b5cab" -relief flat
button .left.browser.pathRow.up -text "上一级" -command browser_up -bg "#eef2f6" -fg "#33485f" -relief flat
pack .left.browser.path -in .left.browser.pathRow -side left -fill x -expand 1 -padx {12 8} -pady 8
pack .left.browser.pathRow.load -side left -padx {0 8} -pady 8
pack .left.browser.pathRow.up -side left -padx {0 12} -pady 8
pack .left.browser.pathRow -fill x
label .left.browser.selected -text "当前未选中文件" -bg white -fg "#5f6f82"
pack .left.browser.selected -anchor w -padx 12 -pady {0 6}
frame .left.browser.listFrame -bg white
scrollbar .left.browser.listScroll -command {.left.browser.list yview}
listbox .left.browser.list -yscrollcommand {.left.browser.listScroll set} -bg "#fbfcfd" -activestyle none -height 16
bind .left.browser.list <<ListboxSelect>> {browser_select}
bind .left.browser.list <Double-1> {browser_open}
pack .left.browser.listScroll -in .left.browser.listFrame -side right -fill y -padx {0 12} -pady {0 12}
pack .left.browser.list -in .left.browser.listFrame -side left -fill both -expand 1 -padx {12 8} -pady {0 12}
pack .left.browser.listFrame -fill both -expand 1

frame .right.preview
style_panel .right.preview
pack .right.preview -fill both -expand 1 -pady {0 12}
add_field_title .right.preview.title "预览画面"
pack .right.preview.title -anchor w -padx 12 -pady {12 6}
frame .right.preview.frame -bg "#0f1720"
pack .right.preview.frame -fill both -expand 1 -padx 12 -pady {0 12}
canvas .right.preview.canvas -bg "#0f1720" -highlightthickness 0 -yscrollcommand {.right.preview.vscroll set} -xscrollcommand {.right.preview.hscroll set}
scrollbar .right.preview.vscroll -orient vertical -command {.right.preview.canvas yview}
scrollbar .right.preview.hscroll -orient horizontal -command {.right.preview.canvas xview}
grid .right.preview.canvas -in .right.preview.frame -row 0 -column 0 -sticky nsew
grid .right.preview.vscroll -in .right.preview.frame -row 0 -column 1 -sticky ns
grid .right.preview.hscroll -in .right.preview.frame -row 1 -column 0 -sticky ew
grid rowconfigure .right.preview.frame 0 -weight 1
grid columnconfigure .right.preview.frame 0 -weight 1

frame .right.summary
style_panel .right.summary
pack .right.summary -fill both -expand 1 -pady {0 12}
add_field_title .right.summary.title "分析摘要"
pack .right.summary.title -anchor w -padx 12 -pady {12 6}
text .right.summary.text -height 14 -relief flat -bg "#f7f9fb" -wrap word
pack .right.summary.text -fill both -expand 1 -padx 12 -pady {0 12}
.right.summary.text insert 1.0 $form(summary)

frame .right.links
style_panel .right.links
pack .right.links -fill x
add_field_title .right.links.title "输出文件"
pack .right.links.title -anchor w -padx 12 -pady {12 6}
button .right.links.json -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
button .right.links.csv -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
button .right.links.video -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
button .right.links.plot -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
button .right.links.card -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
button .right.links.sheet -text "未生成" -state disabled -anchor w -relief flat -bg "#fbfcfd"
pack .right.links.json .right.links.csv .right.links.video .right.links.plot .right.links.card .right.links.sheet \
    -fill x -padx 12 -pady 2

frame .bottom -bg "#eef3f8"
pack .bottom -fill x -padx 14 -pady {0 14}
button .bottom.start -text "开始分析" -command start_analysis -bg "#0b5cab" -fg white -relief flat
button .bottom.stop -text "停止分析" -command stop_analysis -bg "#eef2f6" -fg "#33485f" -relief flat -state disabled
button .bottom.output -text "打开输出目录" -command open_output_dir -bg "#eef2f6" -fg "#33485f" -relief flat
ttk::progressbar .bottom.progress -length 260 -maximum 100
label .bottom.status -textvariable form(status) -bg "#eef3f8" -fg "#24384d" -anchor w
pack .bottom.start -side left
pack .bottom.stop -side left -padx 8
pack .bottom.output -side left -padx 8
pack .bottom.progress -side left -fill x -expand 1 -padx 12
pack .bottom.status -side left

refresh_cameras
update_source_mode
browse_directory $project_root
update_video_preview
wm protocol . WM_DELETE_WINDOW on_close
