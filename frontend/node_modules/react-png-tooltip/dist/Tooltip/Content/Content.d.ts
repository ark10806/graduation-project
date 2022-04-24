import * as React from 'react';
interface IContentProps {
    bIsOpenedByClick?: boolean;
    bIsClickingDisabled?: boolean;
    /**
     * Function that will execute after mounting and before unmounting to calculate position
    /* and handle event listeners
     */
    setTooltip: (handler: string) => void;
    triangleReference: React.RefObject<SVGSVGElement>;
    closeTooltip: () => void;
    className?: string;
    style?: React.CSSProperties;
    reference: React.RefObject<HTMLDivElement>;
    contentReference: React.RefObject<HTMLDivElement>;
    children?: any;
}
declare const content: (props: IContentProps) => JSX.Element | null;
export default content;
