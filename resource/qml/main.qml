import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Window 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15


ApplicationWindow {
    id : root

    width : 1600
    height : 900
    visible : true
    title : qsTr('리덕션 수치모델 기반 창호시공부위 하자 평가 프로그램')

    FontLoader {
        source : '../font/SourceHanSansKR-Regular.otf'
    }
    FontLoader {
        source : '../font/SourceHanSansKR-Medium.otf'
    }
    FontLoader {
        source : '../font/MaterialIconsOutlined-Regular.otf'
    }

    header : TabBar {
        id : tab_bar
        width : parent.width

        background : Rectangle {
            color : Material.primaryColor
        }

        TabButton {
            text : qsTr('File && Option')
            Material.accent : '#ffffff'
        }

        TabButton {
            text : qsTr('Temperature')
            Material.accent : '#ffffff'
        }

        TabButton {
            text : qsTr('Compute')
            Material.accent : '#ffffff'
        }

        TabButton {
            text : qsTr('About')
            Material.accent : '#ffffff'
        }
    }

    footer : StatusBar {
        id : footer
    }

    Page {
        id : page
        anchors.fill : parent
        padding : 10

        StackLayout {
            currentIndex : tab_bar.currentIndex
            anchors.fill : parent

            FileAndOption {
                id : file_option
                Layout.fillHeight : true
                Layout.fillWidth : true
            }

            Temperature {
                id : temperature
                Layout.fillHeight : true
                Layout.fillWidth : true
            }

            RunWidget {
                id : run_widget
                Layout.fillHeight : true
                Layout.fillWidth : true
            }

            About {
                Layout.fillHeight : true
                Layout.fillWidth : true
            }
        }
    }

    CustomPopup {
        id : popup
    }

    function show_popup(title, message, level = 0) {
        popup.title = title
        popup.message = message
        popup.set_icon(level)
        popup.open()
    }

    function progress_bar(active) {
        run_widget.pbar.indeterminate = active
    }

    function status_message(message) {
        footer.status_text.text = message
    }

    function update_model_state(has_matrix, has_model, has_result) {
        run_widget.update_button_hightlights(has_matrix, has_model, has_result)
        footer.update_model_status_icons(has_matrix, has_model)
    }
}
