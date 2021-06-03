import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Rectangle {
    ColumnLayout {
        anchors.fill : parent

        Rectangle {
            width : 200
            height : 300
            Layout.alignment : Qt.AlignHCenter | Qt.AlignVCenter

            ColumnLayout {
                anchors.fill : parent

                Label {
                    Layout.alignment : Qt.AlignHCenter | Qt.AlignVCenter
                    text : '리덕션 수치모델 기반\n창호시공부위 하자 평가 프로그램'
                    font.pointSize : 14
                    font.weight : Font.Medium
                }

                Label {
                    text : 'Version: 0.0.1'
                    font.pointSize : 12
                }

                Label {
                    text : 'Built on 2021-06-03'
                    font.pointSize : 12
                }
            }
        }
    }
}
